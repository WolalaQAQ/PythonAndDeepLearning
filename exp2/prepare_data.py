import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 创建数据集类
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data(data_dir, batch_size, img_size=224, num_workers=4):
    """
    准备数据集

    :param data_dir: 数据集目录
    :param batch_size: 批大小
    :return: DataLoader对象
    """

    # 检查数据集目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录 {data_dir} 不存在。")
    
    # 数据集目录下有train和valid两个子目录，分别对应训练集和验证集
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        raise FileNotFoundError(f"数据集目录 {data_dir} 下缺少 train 或 valid 子目录。")
    
    # train或valid下有多个patient开头的子目录，patient目录下包含以negative和positive结尾的文件夹，里面分别包含阴性和阳性样本的图像
    train_patients = [d for d in os.listdir(train_dir) if d.startswith('patient')]
    valid_patients = [d for d in os.listdir(valid_dir) if d.startswith('patient')]

    if not train_patients or not valid_patients:
        raise FileNotFoundError(f"数据集目录 {data_dir} 下缺少 patient 子目录。")
    
    # 检查每个patient目录下是否有negative和positive子目录
    # for patient in train_patients:
    #     patient_dir = os.path.join(train_dir, patient)
    #     if not os.path.exists(os.path.join(patient_dir, 'negative')) or not os.path.exists(os.path.join(patient_dir, 'positive')):
    #         raise FileNotFoundError(f"患者目录 {patient_dir} 下缺少 negative 或 positive 子目录。")
        
    # for patient in valid_patients:
    #     patient_dir = os.path.join(valid_dir, patient)
    #     if not os.path.exists(os.path.join(patient_dir, 'negative')) or not os.path.exists(os.path.join(patient_dir, 'positive')):
    #         raise FileNotFoundError(f"患者目录 {patient_dir} 下缺少 negative 或 positive 子目录。")
        
    # 读取数据集
    train_data = []
    valid_data = []

    for patient in train_patients:
        patient_dir = os.path.join(train_dir, patient)
        
        for dir in os.listdir(patient_dir):
            dir_path = os.path.join(patient_dir, dir)
            label_name = dir.split('_')[-1]  # negative或positive
            if label_name not in ['negative', 'positive']:
                raise ValueError(f"在目录 {patient_dir} 下的标签 {label_name} 不合法。")
            
            label = 0 if label_name == 'negative' else 1
            for img_file in os.listdir(dir_path):
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
    

    # 对数据做预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    # 创建训练集和验证集的DataLoader
    train_dataset = CustomDataset(train_data, transform=transform)
    valid_dataset = CustomDataset(valid_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader
