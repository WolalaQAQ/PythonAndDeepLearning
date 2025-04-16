from prepare_data import prepare_data
# from net.swin_transformer import SwinTransformer
from net.inceptionv3 import inception_v3, InceptionV3

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
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
from tqdm import tqdm
import toml
from torch.amp import autocast, GradScaler

# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 修改train_epoch函数
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    for inputs, labels in tqdm(dataloader, desc="训练中", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 转换标签格式为浮点型，并调整维度为[batch_size, 1]
        labels_float = labels.float().unsqueeze(1)
        
        # 零梯度
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(enabled=scaler is not None, device_type=device.type, dtype=torch.bfloat16):
            # 前向传播
            if isinstance(model, InceptionV3) and model.training and model.aux_logits:
                outputs, aux_outputs = model(inputs)
                outputs = outputs[:, 1].unsqueeze(1)
                aux_outputs = aux_outputs[:, 1].unsqueeze(1)
                loss = criterion(outputs, labels_float) + 0.4 * criterion(aux_outputs, labels_float)
            else:
                outputs = model(inputs)
                if outputs.shape[1] > 1:
                    outputs = outputs[:, 1].unsqueeze(1)
                loss = criterion(outputs, labels_float)
        
        # 使用GradScaler缩放损失并执行反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 更新统计
        running_loss += loss.item() * inputs.size(0)
        
        # 收集预测和标签用于计算指标
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze()
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(targets, predictions)
    
    return epoch_loss, accuracy

# 修改validate函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="验证中", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            labels_float = labels.float().unsqueeze(1)
            
            # 验证时不需要混合精度
            outputs = model(inputs)
            if isinstance(model, InceptionV3) and model.aux_logits:
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            
            if outputs.shape[1] > 1:
                outputs = outputs[:, 1].unsqueeze(1)
            
            loss = criterion(outputs, labels_float)
            
            # 收集数据用于计算指标
            running_loss += loss.item() * inputs.size(0)
            
            # 获取预测和概率
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().squeeze()
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.squeeze().cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(targets, predictions)
    auc = roc_auc_score(targets, all_probs)
    
    return epoch_loss, accuracy, auc

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
    """根据配置创建模型"""
    model_name = config['model']['model_name']
    
    if model_name == "swin_transformer":
        model = SwinTransformer(
            img_size=config['dataset']['img_size'],
            patch_size=4,
            in_chans=3,
            num_classes=1,  # 改为1个输出，用于二元分类
            embed_dim=96,
            depths=config['model']['depths'],
            num_heads=config['model']['num_heads'],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=config['model']['drop_rate'],
            attn_drop_rate=config['model']['attn_drop_rate'],
            drop_path_rate=config['model']['drop_path_rate'],
            norm_layer=nn.LayerNorm
        )
    elif model_name == "inception_v3":
        model = inception_v3(
            num_classes=2,  # 保持为2，我们会在forward中选择第二个输出
            pretrained=False,
            input_size=config['dataset']['img_size']
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 迁移学习：加载预训练权重（如果提供）
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
    
    # 设置随机种子
    set_seed(config['training']['seed'])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器，获取类别权重
    train_loader, valid_loader, pos_weight = prepare_data(
        config['dataset']['data_dir'], 
        config['dataset']['batch_size'], 
        config['dataset']['img_size'], 
        config['dataset']['num_workers']
    )
    
    # 创建模型
    model = create_model(config, device)
    
    # 将计算得到的正样本权重转换为张量并移至设备
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    
    # 定义损失函数 - 使用加权二元交叉熵，权重基于数据分布自动计算
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"使用加权二元交叉熵损失函数，基于数据分布计算的正样本权重: {pos_weight:.4f}")
    
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
        f.write(f"数据统计:\n")
        f.write(f"  - 基于数据分布自动计算的正样本权重: {pos_weight:.4f}\n")
        if use_amp:
            f.write(f"  - 启用混合精度训练\n\n")
        else:
            f.write("\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练 - 传入scaler
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # 验证
        val_loss, val_acc, val_auc = validate(model, valid_loader, criterion, device)
        
        # 学习率更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 输出结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 记录到实验日志文件
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}/{config['training']['epochs']}\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
            f.write(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}\n")
            f.write(f"  Learning Rate: {current_lr:.6f}\n")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_auc': best_auc,
            }, save_path)
            print(f"保存最佳模型，AUC: {best_auc:.4f}")
            
            # 记录最佳模型信息
            with open(log_file, "a") as f:
                f.write(f"  [NEW BEST] Model saved with AUC: {best_auc:.4f}\n")
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['save_freq'] == 0:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_auc': best_auc,
            }, save_path)
    
    # 训练结束，记录最终性能
    with open(log_file, "a") as f:
        f.write("\n=== Final Results ===\n")
        f.write(f"Best Validation AUC: {best_auc:.4f}\n")
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
    parser = argparse.ArgumentParser(description="训练脚本")
    
    # 添加配置文件路径参数
    parser.add_argument('--config', type=str, default='./cfg/inceptionv3.toml',
                        help='配置文件路径')
    parser.add_argument('--runs_dir', type=str, default='./runs',
                        help='实验结果根目录')
    
    args = parser.parse_args()
    
    # 确保runs目录存在
    os.makedirs(args.runs_dir, exist_ok=True)
    
    main(args)
