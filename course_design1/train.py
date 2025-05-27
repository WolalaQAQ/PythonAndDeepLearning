import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from tqdm import tqdm
import toml
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

# 导入模型和数据准备函数
from net.alexnet import AlexNet
from prepare_data import prepare_cifar10

# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_gpus(config):
    """设置GPU使用环境，尊重配置文件中的设置"""
    # 首先检查系统是否有CUDA设备
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA设备，将使用CPU进行训练")
        return torch.device("cpu")
    
    # 获取系统中可用的GPU数量
    system_gpu_count = torch.cuda.device_count()
    print(f"系统中检测到 {system_gpu_count} 个GPU")
    
    # 检查配置中是否有GPU设置
    if 'hardware' not in config or 'gpus' not in config['hardware']:
        # 默认使用第一个GPU
        print(f"未指定GPU，默认使用GPU 0")
        return torch.device("cuda:0")
    
    gpus = config['hardware']['gpus']
    
    # 如果gpus为空列表或None，使用第一个可用GPU
    if not gpus:
        print(f"未指定具体GPU，将使用GPU 0")
        return torch.device("cuda:0")
    
    # 验证指定的GPU是否超出范围
    valid_gpus = []
    for gpu_id in gpus:
        if gpu_id >= system_gpu_count:
            print(f"警告: 请求的GPU ID {gpu_id} 超出了可用范围(0-{system_gpu_count-1})。")
        else:
            valid_gpus.append(gpu_id)
    
    # 如果没有有效的GPU ID，使用第一个可用的GPU
    if not valid_gpus:
        print(f"警告: 所有指定的GPU ID都不可用，将使用GPU 0")
        return torch.device("cuda:0")
    
    # 始终返回第一个有效的GPU作为主设备
    primary_gpu = valid_gpus[0]
    print(f"设置主GPU为: cuda:{primary_gpu}")
    return torch.device(f"cuda:{primary_gpu}")


def wrap_model_for_multi_gpu(model, config, device):
    """根据配置将模型包装为多GPU模式，仅当配置文件要求使用多个GPU时才包装"""
    if device.type != 'cuda':
        return model
    
    # 检查配置文件中指定的GPU
    if 'hardware' not in config or 'gpus' not in config['hardware']:
        # 配置中未指定GPU，默认使用单GPU
        print("配置中未指定GPU，使用单GPU模式")
        return model
    
    requested_gpus = config['hardware'].get('gpus', [])
    
    # 如果配置中请求的GPU数量为0或1，不进行包装
    if len(requested_gpus) <= 1:
        print(f"配置中只请求了{len(requested_gpus)}个GPU，使用单GPU模式")
        return model
    
    # 检查系统中可用的GPU数量
    system_gpu_count = torch.cuda.device_count()
    
    # 过滤出有效的GPU ID
    valid_gpus = [gpu_id for gpu_id in requested_gpus if gpu_id < system_gpu_count]
    
    # 如果有效GPU数量小于等于1，不进行包装
    if len(valid_gpus) <= 1:
        print(f"配置中请求的多个GPU中，只有{len(valid_gpus)}个有效，使用单GPU模式")
        return model
    
    # 到这里，确认配置请求多GPU且系统确实有多个可用的GPU
    print(f"配置请求使用多个GPU {requested_gpus}，有效GPU: {valid_gpus}")
    print(f"将模型包装为DataParallel模式")
    
    # 使用DataParallel包装模型，只使用配置中指定的有效GPU
    model = DataParallel(model, device_ids=valid_gpus)
    
    return model

# 修改train_epoch函数
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="训练中", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(enabled=scaler is not None, device_type=device.type, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    
    return epoch_loss, accuracy

# 修改validate函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="验证中", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Top-1 准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Top-5 准确率
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    top1_accuracy = correct / total
    top5_accuracy = top5_correct / total
    
    return epoch_loss, top1_accuracy, top5_accuracy

# 加载配置文件
def load_config(config_path):
    """加载TOML配置文件"""
    try:
        config = toml.load(config_path)
        return config
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        exit(1)


# 创建模型
def create_model(config, device):
    """根据配置创建AlexNet模型"""
    model_name = config['model']['model_name']
    
    if model_name == "alexnet":
        # 从配置中获取层归一化参数，默认为False
        use_layer_norm = config['model'].get('use_layer_norm', False)
        model = AlexNet(
            drop_out=config['model']['drop_out'], 
            num_classes=10,  # CIFAR-10有10个类别
            use_layer_norm=use_layer_norm
        )
        print(f"创建AlexNet模型 - 层归一化: {'启用' if use_layer_norm else '禁用'}")
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 如果有预训练权重，加载它们
    if config['model']['pretrained']:
        print(f"加载预训练权重: {config['model']['pretrained']}")
        checkpoint = torch.load(config['model']['pretrained'])
        model.load_state_dict(checkpoint['model'], strict=False)
    
    return model.to(device)


# 主函数
def main(args):
    # 加载配置
    config = load_config(args.config)
    
    # 获取开始时间
    start_time = time.time()
    
    # 自动生成实验ID
    exp_id = get_next_exp_id(args.runs_dir)
    print(f"开始实验 exp_{exp_id}")
    
    # 设置实验目录
    exp_dir = os.path.join(args.runs_dir, f"exp_{exp_id}")
    log_dir = os.path.join(exp_dir, "logs")
    save_dir = os.path.join(exp_dir, "checkpoints")
    
    # 创建必要的目录
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存使用的配置
    with open(os.path.join(exp_dir, "config_used.toml"), 'w') as f:
        toml.dump(config, f)
    
    # 创建实验日志文件
    log_file = os.path.join(exp_dir, "experiment_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Experiment ID: {exp_id}\n")
        f.write(f"Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config['model']['model_name']}\n")
        f.write(f"Parameters:\n")
        for section in config:
            f.write(f"  [{section}]\n")
            for key, value in config[section].items():
                f.write(f"    {key}: {value}\n")
        f.write("\n=== Training Progress ===\n")
    
    # 设置GPU
    try:
        device = setup_gpus(config)
        print(f"主设备: {device}")
    except Exception as e:
        print(f"GPU设置出错，将使用CPU: {e}")
        device = torch.device("cpu")
    
    # 设置随机种子
    set_seed(config['training']['seed'])
    
    # 创建数据集和数据加载器 - 修改为CIFAR-10专用函数
    try:
        train_loader, valid_loader = prepare_cifar10(config)
        print("成功加载数据，使用完整的数据增强")
    except OverflowError as e:
        print(f"数据增强参数导致溢出错误: {e}")
        print("尝试使用保守的数据增强参数...")
        # 创建一个修改后的配置，禁用可能导致溢出的数据增强
        modified_config = config.copy()
        if 'data' not in modified_config:
            modified_config['data'] = {}
        modified_config['data']['use_data_augmentation'] = False
        modified_config['data']['num_workers'] = 0
        try:
            train_loader, valid_loader = prepare_cifar10(modified_config)
            print("成功加载数据，已禁用数据增强")
        except Exception as e2:
            print(f"禁用数据增强后仍有错误: {e2}")
            # 最后的降级方案：使用最基本的配置
            basic_config = {
                'data': {
                    'batch_size': 32,
                    'num_workers': 0,
                    'use_data_augmentation': False,
                    'pin_memory': False
                }
            }
            train_loader, valid_loader = prepare_cifar10(basic_config)
            print("使用基本配置成功加载数据")
    except Exception as e:
        print(f"数据加载出错: {e}")
        print("尝试使用更保守的数据加载参数...")
        # 创建一个修改后的配置，设置 num_workers=0
        modified_config = config.copy()
        if 'data' not in modified_config:
            modified_config['data'] = {}
        modified_config['data']['num_workers'] = 0
        modified_config['data']['use_data_augmentation'] = False
        try:
            train_loader, valid_loader = prepare_cifar10(modified_config)
            print("成功加载数据，已禁用数据增强和多进程")
        except Exception as e2:
            print(f"保守配置仍有错误: {e2}")
            # 最后的降级方案
            basic_config = {
                'data': {
                    'batch_size': 16,
                    'num_workers': 0,
                    'use_data_augmentation': False,
                    'pin_memory': False
                }
            }
            train_loader, valid_loader = prepare_cifar10(basic_config)
            print("使用最基本配置成功加载数据")

    # 创建模型
    model = create_model(config, device)
    
    # 根据配置决定是否使用多GPU
    try:
        model = wrap_model_for_multi_gpu(model, config, device)
    except Exception as e:
        print(f"多GPU包装出错: {e}")
        print("将继续使用单GPU")
    
    # 定义损失函数 - 多分类使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    print("使用交叉熵损失函数进行多分类任务")
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs'], 
        eta_min=config['training']['min_lr']
    )
    
    # 创建GradScaler用于混合精度训练
    use_amp = device.type == 'cuda' and config['training'].get('use_amp', True)
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("启用混合精度训练以提高速度")
    else:
        print("未启用混合精度训练")
    
    # 记录训练配置到日志
    with open(log_file, "a") as f:
        f.write(f"数据集: CIFAR-10 (10类分类)\n")
        if use_amp:
            f.write(f"启用混合精度训练\n\n")
        else:
            f.write("\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # 验证
        val_loss, val_top1_acc, val_top5_acc = validate(model, valid_loader, criterion, device)
        
        # 学习率更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 输出结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Top-1 Acc: {val_top1_acc:.4f}, Val Top-5 Acc: {val_top5_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val_top1', val_top1_acc, epoch)
        writer.add_scalar('Accuracy/val_top5', val_top5_acc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 记录到实验日志文件
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}/{config['training']['epochs']}\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Val Top-1 Acc: {val_top1_acc:.4f}, Val Top-5 Acc: {val_top5_acc:.4f}\n")
            f.write(f"  Learning Rate: {current_lr:.6f}\n")
        
        # 保存最佳模型（基于Top-1准确率）
        if val_top1_acc > best_acc:
            best_acc = val_top1_acc
            save_path = os.path.join(save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f"保存最佳模型，Top-1 Acc: {best_acc:.4f}")
            
            # 记录最佳模型信息
            with open(log_file, "a") as f:
                f.write(f"  [NEW BEST] Model saved with Top-1 Acc: {best_acc:.4f}\n")
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['save_freq'] == 0:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, save_path)
    
    # 训练结束，记录最终性能
    with open(log_file, "a") as f:
        f.write("\n=== Final Results ===\n")
        f.write(f"Best Validation Top-1 Accuracy: {best_acc:.4f}\n")
        f.write(f"Total Training Time: {time.time() - start_time:.2f} seconds\n")
    
    writer.close()
    print("训练完成！")


# 添加一个函数用于自动生成实验ID
def get_next_exp_id(runs_dir):
    """
    根据runs目录下已有的实验文件夹自动生成下一个实验ID
    返回格式为简单数字字符串
    """
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        return "1"
    
    existing_exps = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("exp_")]
    
    if not existing_exps:
        return "1"
    
    # 提取现有实验的数字部分并找到最大值
    exp_numbers = []
    for exp in existing_exps:
        try:
            # 从exp_X中提取X部分并转为整数
            exp_num = int(exp.split("_")[1])
            exp_numbers.append(exp_num)
        except (IndexError, ValueError):
            continue
    
    if not exp_numbers:
        return "1"
    
    # 生成下一个实验ID
    next_exp_num = max(exp_numbers) + 1
    return str(next_exp_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 AlexNet训练脚本")
    
    parser.add_argument('--config', type=str, default='./cfg/alexnet.toml',
                        help='配置文件路径')
    parser.add_argument('--runs_dir', type=str, default='./runs',
                        help='实验结果根目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.runs_dir, exist_ok=True)
    
    main(args)
