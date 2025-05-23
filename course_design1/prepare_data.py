import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import psutil  # 用于检查系统内存情况
import toml
import random

# 修改CustomDataset以支持CIFAR-10格式的数据
class CustomDataset(Dataset):
    def __init__(self, data, config, transform=None, use_cache=True, verbose=True):
        self.data = data  # 对于CIFAR-10，这将是(image, label)的列表
        self.config = config
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
        
        # 这是一个粗略估计，实际内存使用会因图像尺寸和格式而异
        est_image_size = self.config['dataset']['img_size'] * self.config['dataset']['img_size'] * 3  # 基本估计值，单位：字节
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
            print("正在将CIFAR-10图像缓存到内存中...")
        
        # 使用tqdm显示进度
        for idx, (image, _) in enumerate(tqdm(self.data, desc="缓存图像", disable=not self.verbose)):
            # 对于CIFAR-10，图像已经是PIL Image或numpy array
            if isinstance(image, np.ndarray):
                # 如果是numpy array，转换为PIL Image
                image = Image.fromarray(image)
            self.cache[idx] = image
        
        if self.verbose:
            print(f"已完成图像缓存，共 {len(self.cache)} 张图像")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 从缓存中获取图像或从数据中直接获取
        if self.use_cache and idx in self.cache:
            image = self.cache[idx]
        else:
            image, _ = self.data[idx]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        
        _, label = self.data[idx]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 定义可以被pickle的辅助函数
class IdentityTransform:
    """恒等变换，用于替代lambda函数"""
    def __call__(self, x):
        return x

class ConditionalGaussianBlur:
    """条件高斯模糊变换"""
    def __init__(self, kernel_size, sigma_range, prob):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.prob = prob
        self.blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma_range)
        self.identity = IdentityTransform()
    
    def __call__(self, x):
        if random.random() < self.prob:
            return self.blur_transform(x)
        else:
            return self.identity(x)

def prepare_cifar10(config):
    """
    准备CIFAR-10数据集，使用CustomDataset以支持缓存功能

    :return: train_loader, valid_loader
    """
    # 首先下载原始CIFAR-10数据集
    print("下载CIFAR-10数据集...")
    original_train = datasets.CIFAR10(
        root=config['dataset']['data_dir'], 
        train=True, 
        download=True, 
        transform=None  # 先不应用变换
    )
    
    original_valid = datasets.CIFAR10(
        root=config['dataset']['data_dir'], 
        train=False, 
        download=True, 
        transform=None  # 先不应用变换
    )
    
    # 将数据转换为CustomDataset可以使用的格式
    print("准备训练数据...")
    train_data = []
    for i in range(len(original_train)):
        image, label = original_train[i]
        train_data.append((image, label))
    
    print("准备验证数据...")
    valid_data = []
    for i in range(len(original_valid)):
        image, label = original_valid[i]
        valid_data.append((image, label))
    
    # 获取数据变换
    img_size = config['dataset']['img_size']
    
    # 创建训练数据的增强变换列表
    train_transforms = [
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.RandomHorizontalFlip(p=config['augmentation']['h_flip_prob']),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=config['augmentation']['v_flip_prob']),    # 随机垂直翻转
        transforms.RandomRotation(config['augmentation']['rotation_degrees']),     # 随机旋转
    ]
    
    # 条件添加随机裁剪
    if config['augmentation']['random_crop']:
        train_transforms.append(transforms.RandomResizedCrop(
            size=img_size,
            scale=(config['augmentation']['crop_scale_min'], 1.0),
            ratio=(0.75, 1.33),
            antialias=True
        ))
    
    # 条件添加高斯模糊
    if config['augmentation']['blur_prob'] > 0:
        train_transforms.append(ConditionalGaussianBlur(
            kernel_size=config['augmentation']['blur_kernel_size'],
            sigma_range=(0.1, 2.0),
            prob=config['augmentation']['blur_prob']
        ))
    
    # 添加其他变换
    train_transforms.extend([
        transforms.RandomGrayscale(p=config['augmentation']['grayscale_prob']),  # 随机灰度转换
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
        # CIFAR-10的标准化参数
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    
    # 条件添加随机擦除
    if config['augmentation']['use_random_erase']:
        train_transforms.append(transforms.RandomErasing(
            p=config['augmentation']['erase_prob'],
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        ))
    
    # 组合所有训练变换
    train_transform = transforms.Compose(train_transforms)
    
    # 验证集只需要调整大小和标准化，不需要增强
    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # CIFAR-10的标准化参数
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # 使用CustomDataset创建数据集
    use_cache = config['dataset']['use_cache']
    train_dataset = CustomDataset(
        train_data, 
        config, 
        transform=train_transform, 
        use_cache=use_cache,
        verbose=True
    )
    
    valid_dataset = CustomDataset(
        valid_data, 
        config, 
        transform=valid_transform, 
        use_cache=use_cache,
        verbose=True
    )
    
    # 创建数据加载器
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    
    # 如果使用了缓存，可以减少num_workers，因为IO不再是瓶颈
    if use_cache:
        actual_workers = min(2, num_workers)  # 使用较少的工作线程
        print(f"启用缓存功能，调整工作线程数量为 {actual_workers}")
    else:
        actual_workers = num_workers
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=actual_workers, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=actual_workers, 
        pin_memory=True
    )
    
    return train_loader, valid_loader

# 保留prepare_data函数作为通用接口
def prepare_data(config):
    """通用数据准备接口，重定向到CIFAR-10"""
    return prepare_cifar10(config)