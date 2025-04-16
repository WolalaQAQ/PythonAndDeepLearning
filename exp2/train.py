from prepare_data import prepare_data
from net.swin_transformer import SwinTransformer

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


# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    for inputs, labels in tqdm(dataloader, desc="训练中", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 更新统计
        running_loss += loss.item() * inputs.size(0)
        
        # 收集预测和标签用于计算指标
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(targets, predictions)
    
    return epoch_loss, accuracy
# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 收集数据用于计算指标
            running_loss += loss.item() * inputs.size(0)
            
            # 获取预测和概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 异常类别的概率
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(targets, predictions)
    auc = roc_auc_score(targets, all_probs)
    
    return epoch_loss, accuracy, auc

# 主函数
def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    train_loader, valid_loader = prepare_data(args.data_dir, args.batch_size, args.img_size, args.num_workers)
    
    # 创建Swin Transformer模型
    model = SwinTransformer(
        img_size=args.img_size,
        patch_size=4,
        in_chans=3,
        num_classes=2,  # 二分类：正常/异常
        embed_dim=96,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        norm_layer=nn.LayerNorm
    )
    
    # 迁移学习：使用ImageNet预训练权重（如果需要）
    if args.pretrained:
        print("加载预训练权重...")
        # 注意：需要处理权重加载和可能的不匹配情况
        # 这里仅作为示例
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model'], strict=False)
    
    model = model.to(device)
    
    # 定义损失函数
    # 如果数据集类别不平衡，可以使用加权交叉熵
    if args.weighted_loss:
        # 计算类别权重
        class_counts = [0, 0]
        for _, label in train_dataset:
            class_counts[label] += 1
        
        total = sum(class_counts)
        class_weights = [total / (len(class_counts) * count) for count in class_counts]
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权损失函数，权重: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
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
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_auc': best_auc,
            }, save_path)
            print(f"保存最佳模型，AUC: {best_auc:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_auc': best_auc,
            }, save_path)
    
    writer.close()
    print("训练完成！")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练脚本")
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./data/elbow', 
                        help='数据集根目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图像大小')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载器的工作线程数')
    
    # 模型相关参数
    parser.add_argument('--depths', type=list, default=[2, 2, 6, 2], 
                        help='每层的Transformer块数')
    parser.add_argument('--num_heads', type=list, default=[3, 6, 12, 24], 
                        help='每层的注意力头数')
    parser.add_argument('--drop_rate', type=float, default=0.0, 
                        help='Dropout比率')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, 
                        help='注意力Dropout比率')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, 
                        help='Drop path比率')
    parser.add_argument('--pretrained', type=str, default='', 
                        help='预训练模型路径')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100, 
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05, 
                        help='权重衰减')
    parser.add_argument('--weighted_loss', action='store_true', default=False,
                        help='是否使用加权损失函数（处理类别不平衡）')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='TensorBoard日志目录')
    parser.add_argument('--save_freq', type=int, default=10, 
                        help='保存检查点的频率（轮数）')
    
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)