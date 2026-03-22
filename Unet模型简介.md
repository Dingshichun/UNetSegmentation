# unet 模型

unet 的核心结构包括：  
* **编码器**：一系列卷积池化层，用于捕获图像的上下文信息。
* **解码器**：系列上采样层和卷积层，用于恢复分割的精细细节。
* **跳跃连接**：从收缩路径复制特征图到相应的扩展路径，以保留更多的细节信息。

DiceBCELoss损失函数，是医学图像分割领域最常用的复合损失函数之一。

## 一、DiceBCELoss的定义

**DiceBCELoss** 是 **Dice Loss** 和 **Binary Cross Entropy (BCE) Loss** 的加权和，结合了两种损失函数的优势：

$$\mathcal{L}_{DiceBCE} = \alpha \cdot \mathcal{L}_{Dice} + \beta \cdot \mathcal{L}_{BCE}$$

通常设置 $\alpha = 1$, $\beta = 1$，即简单相加：
$$\mathcal{L}_{DiceBCE} = \mathcal{L}_{Dice} + \mathcal{L}_{BCE}$$

---

## 二、组成成分详解

### 1. Dice Loss（Sørensen-Dice系数）

源自Dice相似系数，衡量预测与真实标签的重叠程度：

$$\mathcal{L}_{Dice} = 1 - \frac{2|X \cap Y|}{|X| + |Y|} = 1 - \frac{2\sum_{i} p_i g_i}{\sum_{i} p_i^2 + \sum_{i} g_i^2}$$

| 特性 | 说明 |
|------|------|
| **范围** | $[0, 1]$，0表示完美匹配 |
| **核心优势** | 对类别不平衡（小目标）鲁棒 |
| **主要问题** | 梯度不稳定，极端情况下可能梯度消失 |

**平滑版本**（防止除零）：
$$\mathcal{L}_{Dice} = 1 - \frac{2\sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}$$

### 2. Binary Cross Entropy Loss

标准的像素级分类损失：

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i}\left[ g_i \log(p_i) + (1-g_i)\log(1-p_i) \right]$$

| 特性 | 说明 |
|------|------|
| **范围** | $[0, +\infty)$ |
| **核心优势** | 梯度稳定，训练初期收敛快 |
| **主要问题** | 对类别不平衡敏感，可能忽略小目标 |

---

## 三、为什么需要结合？

```
┌─────────────────────────────────────────────────────────┐
│                    单独使用的问题                          │
├─────────────────────────────────────────────────────────┤
│  仅用Dice Loss          │  仅用BCE Loss                   │
│  • 初期训练不稳定         │  • 小目标被背景淹没              │
│  • 梯度可能消失         │  • 边界分割模糊                 │
│  • 收敛速度慢            │  • 对不平衡数据效果差            │
├─────────────────────────────────────────────────────────┤
│                    DiceBCELoss的优势                     │
│  ✓ Dice处理类别不平衡  +  BCE提供稳定梯度                  │
│  ✓ 小目标检测能力强    +  训练初期快速收敛                  │
│  ✓ 边界分割更清晰      +  整体训练更稳定                    │
└─────────────────────────────────────────────────────────┘
```

---

## 四、PyTorch实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, bce_weight=1.0, dice_weight=1.0):
        """
        Args:
            smooth: 平滑因子，防止除零
            bce_weight: BCE损失的权重
            dice_weight: Dice损失的权重
        """
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()  # 内置sigmoid的BCE
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 网络原始输出 (未经过sigmoid)，shape: [N, 1, H, W]
            targets: 真实标签，shape: [N, 1, H, W]，值为0或1
        """
        # BCE Loss（使用内置sigmoid）
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss（手动应用sigmoid）
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # 展平计算
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score
        
        # 组合损失
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


# ============ 使用示例 ============
if __name__ == "__main__":
    # 模拟数据
    batch_size, height, width = 4, 256, 256
    
    # 网络输出（logits，未归一化）
    predictions = torch.randn(batch_size, 1, height, width)
    
    # 真实标签（二值）
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # 初始化损失函数
    criterion = DiceBCELoss(bce_weight=1.0, dice_weight=1.0)
    
    # 计算损失
    loss = criterion(predictions, targets)
    print(f"DiceBCE Loss: {loss.item():.4f}")
```

---

## 五、进阶变体与优化技巧

### 1. Focal-DiceBCE Loss（处理极端不平衡）

```python
class FocalDiceBCELoss(nn.Module):
    """加入Focal Loss思想，降低易分样本权重"""
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.focal = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Focal-weighted BCE
        bce = self.focal(inputs, targets)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_bce = (focal_weight * bce).mean()
        
        # Dice Loss
        dice = self._dice_loss(torch.sigmoid(inputs), targets)
        
        return focal_bce + dice
    
    def _dice_loss(self, pred, target):
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
```

### 2. 多类别分割版本（Softmax DiceBCE）

```python
class MultiClassDiceBCELoss(nn.Module):
    """适用于多类别分割（如UNet++）"""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, inputs, targets):
        # inputs: [N, C, H, W], targets: [N, H, W] 类别索引
        
        # CrossEntropy Loss
        ce_loss = self.ce(inputs, targets)
        
        # Multi-class Dice Loss（每个类别计算后平均）
        probs = F.softmax(inputs, dim=1)
        dice_loss = 0
        
        for c in range(self.num_classes):
            pred_c = probs[:, c, ...]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            dice_c = (2 * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_loss += (1 - dice_c)
        
        return ce_loss + dice_loss / self.num_classes
```

### 3. 权重调优策略

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| **标准医学图像** | `dice=1.0, bce=1.0` | 平衡组合 |
| **极小目标**（如病灶<1%） | `dice=2.0, bce=0.5` | 增强Dice比重 |
| **边界模糊** | `dice=1.5, bce=1.0` + 边界损失 | 增强结构感知 |
| **训练初期震荡** | `dice=0.5, bce=1.5` → 逐步调整 | 先稳定再优化 |

---

## 六、可视化理解

```
真实标签 (Ground Truth)          预测输出 (Prediction)
┌─────────────┐                ┌─────────────┐
│  ░░░░░░░░   │                │  ▓▓▓▓▓▓▓▓   │
│  ░░████░░   │      vs       │  ▓▓████▓▓   │
│  ░░████░░   │                │  ▓▓██████   │
│  ░░░░░░░░   │                │  ▓▓▓▓▓▓▓▓   │
└─────────────┘                └─────────────┘

BCE关注：每个像素是否正确分类（像素级）
         → 边缘像素可能被误判为背景

Dice关注：整体形状重叠度（区域级）  
         → 强制预测区域与真实区域形状匹配

DiceBCE = 像素精度 + 结构相似性 的双重约束
```

---

## 七、总结对比表

| 损失函数 | 适用场景 | 对小目标 | 梯度稳定性 | 收敛速度 |
|---------|---------|---------|-----------|---------|
| BCE Only | 简单二分类 | ❌ 差 | ✅ 稳定 | ⚡ 快 |
| Dice Only | 类别平衡的分割 | ✅ 好 | ❌ 不稳定 | 🐢 慢 |
| **DiceBCE** | **医学图像分割** | **✅ 优秀** | **✅ 稳定** | **⚡ 适中** |
| Tversky | 极度不平衡 | ✅✅ 极好 | ⚠️ 需调参 | 🐢 较慢 |
| Focal | 前景背景悬殊 | ✅ 好 | ⚠️ 敏感 | ⚡ 快 |

**DiceBCELoss是医学图像分割（如UNet、UNet++、DeepLab等）的默认首选损失函数**，特别是在肿瘤分割、器官分割等类别不平衡场景中表现优异。