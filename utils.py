import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import segmentation_models_pytorch as smp
import os
from PIL import Image
from tqdm import tqdm

class PlantVillageDataset(Dataset):
    """PlantVillage分割数据集"""
    
    def __init__(self, root_dir, split='train', transform=None, img_size=256):
        """
        参数：
            root_dir: 数据集根目录（包含train/val/test子目录）
            split: 'train', 'val', 或 'test'
            transform: 数据增强变换
            img_size: 图像大小
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # 获取图像和掩模路径
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        
        # 确保目录存在
        if not os.path.exists(self.image_dir):
            raise ValueError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"掩模目录不存在: {self.mask_dir}")
        
        # 获取所有图像文件
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                  if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))])
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # 图像归一化（使用ImageNet统计量）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"{split}数据集加载完成，共 {len(self.image_files)} 个样本")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 加载掩模（尝试不同扩展名）
        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        
        # 尝试查找掩模文件
        for ext in ['.png', '.jpg', '.PNG', '.JPG']:
            potential_path = os.path.join(self.mask_dir, base_name+"_final_masked" + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"找不到掩模文件: {base_name}")
        
        # 使用PIL加载
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度
        
        # 应用基础变换
        image = self.base_transform(image)
        mask = self.base_transform(mask)
        
        # 数据增强（仅训练集）
        if self.transform and self.split == 'train':
            # 合并图像和掩模进行相同的空间变换
            stacked = torch.cat([image, mask], dim=0)
            stacked = self.transform(stacked)
            image = stacked[:3, :, :]
            mask = stacked[3:, :, :]
        
        # 图像归一化
        image = self.normalize(image)
        
        # 将掩模二值化（假设掩模中非黑色区域为前景）
        mask = (mask > 0.1).float()
        
        return image, mask

class JointTransform:
    """同时对图像和掩模应用相同的数据增强"""
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        # x: [4, H, W] 其中前3通道是图像，第4通道是掩模
        if torch.rand(1) < self.p:
            # 随机水平翻转
            x = transforms.functional.hflip(x)
        
        if torch.rand(1) < self.p:
            # 随机垂直翻转
            x = transforms.functional.vflip(x)
        
        if torch.rand(1) < self.p:
            # 随机旋转（-10到10度）
            angle = torch.rand(1).item() * 20 - 10
            x = transforms.functional.rotate(x, angle)
        
        return x

def create_mobilenet_unet(num_classes=1):
    """
    创建MobileNetV2作为编码器的UNet模型
    
    参数：
        num_classes: 输出类别数（分割任务通常为1或类别数）
    
    返回：
        PyTorch模型
    """
    # 直接使用 encoder_weights="imagenet" 加载权重配置文件时，
    # 我无法打开https://huggingface.co/smp-hub/mobilenet_v2.imagenet/resolve/e67aa804e17f7b404b629127eabbd224c4e0690b/config.json
    # 最后还是会尝试从 torchvision 加载预训练权重，所以直接从 torchvision 加载，
    # 然后再将权重复制到 smp 模型的编码器，详细实现见注释下方
    # model = smp.Unet(
    #     encoder_name="mobilenet_v2",      # MobileNetV2编码器
    #     encoder_weights="imagenet",       # ImageNet预训练权重
    #     in_channels=3,                    # 输入通道数（RGB）
    #     classes=num_classes,              # 输出类别数
    #     activation=None,                  # 训练时不加激活，在损失函数中处理
    #     decoder_channels=(256, 128, 64, 32, 16),  # 轻量解码器通道数
    # )
    
    # 1. 先从 torchvision 加载预训练权重（通常能成功，因为权重已使用 pip 安装，如果没安装，会从pytorch官方下载）
    # IMAGENET1K 表示这是用 ImageNet-1K 数据集​ 训练的权重
    # 该数据集包含 1000 个类别，大约 128 万张训练图片
    # V1 表示这是该模型的 第一个版本​ 的预训练权重
    # 不同版本的权重在训练策略、数据增强、超参数上可能有差异
    mobilenet_pretrained = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # 2. 创建 SMP 模型，但不从网络加载权重
    model = smp.Unet(
        encoder_name="mobilenet_v2", 
        encoder_weights=None,  # 设置为 None，不自动加载
        in_channels=3,
        classes=num_classes,
        activation=None,
        decoder_channels=(256, 128, 64, 32, 16), # 轻量解码器通道数
    )

    # 3. 手动将 torchvision 权重复制到 SMP 编码器
    # SMP 的编码器结构与 torchvision 兼容
    model.encoder.load_state_dict(mobilenet_pretrained.state_dict())

    print("成功从本地 torchvision 加载预训练权重！")
    
    # 冻结编码器前几层（可选，用于迁移学习）
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
    return model

class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss组合"""
    
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
        
    def forward(self, inputs, targets, smooth=1):
        # BCE损失
        bce = self.bce_loss(inputs, targets)
        
        # Dice损失
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # 组合损失
        return bce + (1 - dice)

def calculate_iou(pred, target, threshold=0.5):
    """计算IoU（交并比）"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    
    # 避免除零
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算IoU
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            batch_iou = calculate_iou(preds, masks)
            total_iou += batch_iou.item()
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
            'IoU': f'{batch_iou:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou

def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 计算IoU
            preds = torch.sigmoid(outputs)
            batch_iou = calculate_iou(preds, masks)
            
            total_loss += loss.item()
            total_iou += batch_iou.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                'IoU': f'{batch_iou:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou
