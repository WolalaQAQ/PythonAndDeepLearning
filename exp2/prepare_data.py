import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import psutil  # 用于检查系统内存情况
import toml
import random

# 创建支持缓存的数据集类
class CustomDataset(Dataset):
    def __init__(self, data, transform=None, use_cache=True, verbose=True):
        self.data = data
        self.transform = transform
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache = {}  # 用于存储缓存的图像
        
        # 预加载数据到内存
        if self.use_cache:
            self._cache_images()
    
    def _cache_images(self):
        """预加载所有图像到内存中"""
        # 检查可用内存
        available_memory = psutil.virtual_memory().available
        
        # 估计所需内存（假设每张RGB图像尺寸为224x224，每个像素3字节）
        # 这是一个粗略估计，实际内存使用会因图像尺寸和格式而异
        est_image_size = 320 * 320 * 3  # 基本估计值，单位：字节
        est_total_size = est_image_size * len(self.data)
        
        # 如果估计的缓存大小超过可用内存的70%，发出警告
        if est_total_size > 0.7 * available_memory:
            print(f"警告：缓存整个数据集可能需要约 {est_total_size / (1024**3):.2f} GB，"
                  f"而系统可用内存仅有 {available_memory / (1024**3):.2f} GB")
            proceed = input("是否继续缓存数据集？(y/n): ").lower() == 'y'
            if not proceed:
                self.use_cache = False
                print("已禁用数据集缓存功能")
                return
        
        if self.verbose:
            print("正在将图像缓存到内存中...")
        
        # 使用tqdm显示进度
        for idx, (img_path, _) in enumerate(tqdm(self.data, desc="缓存图像", disable=not self.verbose)):
            # 读取并缓存原始图像
            image = Image.open(img_path).convert('RGB')
            self.cache[idx] = image
        
        if self.verbose:
            print(f"已完成图像缓存，共 {len(self.cache)} 张图像")
            # 报告内存使用情况
            memory_used = sum(image.size[0] * image.size[1] * 3 for image in self.cache.values())
            print(f"估计缓存使用内存: {memory_used / (1024**3):.2f} GB")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        _, label = self.data[idx]
        
        # 从缓存中获取图像或从磁盘读取
        if self.use_cache and idx in self.cache:
            image = self.cache[idx]
        else:
            img_path, _ = self.data[idx]
            image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label

def prepare_data(config):
    """
    准备数据集并计算类别权重，支持数据缓存
    :param data_dir: 数据集目录
    :param batch_size: 批大小
    :param img_size: 图像大小
    :param num_workers: 数据加载线程数
    :param use_cache: 是否将数据集缓存到内存中
    :return: train_loader, valid_loader, pos_weight
    """

    # 检查数据集目录是否存在
    data_dir = config['dataset']['data_dir']
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录 {data_dir} 不存在。")
    
    # 数据集目录下有train和valid两个子目录，分别对应训练集和验证集
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        raise FileNotFoundError(f"数据集目录 {data_dir} 下缺少 train 或 valid 子目录。")
    
    # train或valid下有多个patient开头的子目录
    train_patients = [d for d in os.listdir(train_dir) if d.startswith('patient')]
    valid_patients = [d for d in os.listdir(valid_dir) if d.startswith('patient')]

    if not train_patients or not valid_patients:
        raise FileNotFoundError(f"数据集目录 {data_dir} 下缺少 patient 子目录。")
    
    # 读取数据集
    train_data = []
    valid_data = []
    
    # 统计正负样本数量，用于计算权重
    neg_count = 0
    pos_count = 0

    for patient in train_patients:
        patient_dir = os.path.join(train_dir, patient)
        
        for dir in os.listdir(patient_dir):
            dir_path = os.path.join(patient_dir, dir)
            label_name = dir.split('_')[-1]  # negative或positive
            if label_name not in ['negative', 'positive']:
                raise ValueError(f"在目录 {patient_dir} 下的标签 {label_name} 不合法。")
            
            label = 0 if label_name == 'negative' else 1
            files_in_dir = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            
            # 增加样本计数
            if label == 0:
                neg_count += len(files_in_dir)
            else:
                pos_count += len(files_in_dir)
                
            for img_file in files_in_dir:
                img_path = os.path.join(dir_path, img_file)
                train_data.append((img_path, label))
            
    for patient in valid_patients:
        patient_dir = os.path.join(valid_dir, patient)
        
        for dir in os.listdir(patient_dir):
            dir_path = os.path.join(patient_dir, dir)
            label_name = dir.split('_')[-1]
            if label_name not in ['negative', 'positive']:
                raise ValueError(f"在目录 {patient_dir} 下的标签 {label_name} 不合法。")
            
            label = 0 if label_name == 'negative' else 1
            for img_file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_file)
                valid_data.append((img_path, label))

    # 检查数据集是否为空
    if not train_data:
        raise ValueError("训练集为空，请检查数据集。")
    if not valid_data:
        raise ValueError("验证集为空，请检查数据集。")
    
    # 计算正样本权重 (适用于BCEWithLogitsLoss)
    # 公式: negative_samples / positive_samples
    if pos_count == 0:
        raise ValueError("训练集中没有正样本，无法计算类别权重。")
    
    # 为BCEWithLogitsLoss计算pos_weight
    pos_weight = neg_count / pos_count
    
    print(f"数据集统计：")
    print(f"  - 负样本数量: {neg_count}")
    print(f"  - 正样本数量: {pos_count}")
    print(f"  - 正样本权重: {pos_weight:.4f}")

    # 对数据做预处理
    img_size = config['dataset']['img_size']
    
    # 创建训练数据的增强变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.RandomHorizontalFlip(p=config['augmentation']['h_flip_prob']),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=config['augmentation']['v_flip_prob']),    # 随机垂直翻转
        transforms.RandomRotation(config['augmentation']['rotation_degrees']),     # 随机旋转
        
        # 新增：随机裁剪并调整回原始大小
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(config['augmentation']['crop_scale_min'], 1.0),
            ratio=(0.75, 1.33),
            antialias=True
        ) if config['augmentation']['random_crop'] else transforms.Lambda(lambda x: x),
        
        # 新增：随机应用高斯模糊
        transforms.GaussianBlur(
            kernel_size=config['augmentation']['blur_kernel_size'],
            sigma=(0.1, 2.0)
        ) if random.random() < config['augmentation']['blur_prob'] else transforms.Lambda(lambda x: x),
        
        # 新增：随机灰度转换
        transforms.RandomGrayscale(p=config['augmentation']['grayscale_prob']),
        
        transforms.ColorJitter(
            brightness=config['augmentation']['brightness'],
            contrast=config['augmentation']['contrast'],
            saturation=config['augmentation']['saturation'],
            hue=config['augmentation']['hue']
        ),  # 颜色抖动
        transforms.RandomAffine(
            degrees=0,
            translate=(config['augmentation']['translate'], config['augmentation']['translate']),
            scale=(1-config['augmentation']['scale'], 1+config['augmentation']['scale']),
        ),  # 随机仿射变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 新增：随机擦除
        transforms.RandomErasing(
            p=config['augmentation']['erase_prob'],
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        ) if config['augmentation']['use_random_erase'] else transforms.Lambda(lambda x: x),
    ])
    
    # 验证集只需要调整大小，不需要增强
    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    use_cache = config['dataset']['use_cache']
    # 创建训练集和验证集的DataLoader，使用不同的transform
    train_dataset = CustomDataset(train_data, transform=train_transform, use_cache=use_cache)
    valid_dataset = CustomDataset(valid_data, transform=valid_transform, use_cache=use_cache)
    
    # 如果使用了缓存，可以减少num_workers，因为IO不再是瓶颈
    num_workers = config['dataset']['num_workers']
    if use_cache:
        actual_workers = min(2, num_workers)  # 使用较少的工作线程
    else:
        actual_workers = num_workers
    
    batch_size = config['dataset']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=actual_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=actual_workers, pin_memory=True)
    
    return train_loader, valid_loader, pos_weight
