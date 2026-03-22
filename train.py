import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from utils import * 

def main():
    # 超参数配置
    config = {
        'data_dir': './PlantVillage_processed',  # 处理后的数据集路径
        'img_size': 256,                               # 图像大小
        'batch_size': 8,                               # 批大小
        'num_epochs': 6,                              # 训练轮数
        'learning_rate': 1e-4,                         # 学习率
        'num_workers': 4,                              # 数据加载线程数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',                     # 模型保存目录
        'num_classes': 1,                              # 分割类别数（二分类）
    }
    
    print(f"使用设备: {config['device']}")
    print(f"配置参数: {config}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ========== 数据加载 ==========
    print("\n" + "="*50)
    print("加载数据集...")
    
    # 训练集数据增强
    train_transform = JointTransform(p=0.5)
    
    # 创建数据集
    train_dataset = PlantVillageDataset(
        root_dir=config['data_dir'],
        split='train',
        transform=train_transform,
        img_size=config['img_size']
    )
    
    val_dataset = PlantVillageDataset(
        root_dir=config['data_dir'],
        split='val',
        transform=None,  # 验证集不需要数据增强
        img_size=config['img_size']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    
    # ========== 模型初始化 ==========
    print("\n" + "="*50)
    print("初始化模型...")
    
    model = create_mobilenet_unet(num_classes=config['num_classes'])
    model = model.to(config['device'])
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # ========== 损失函数和优化器 ==========
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ========== 训练循环 ==========
    print("\n" + "="*50)
    print("开始训练...")
    
    best_val_iou = 0.0
    train_history = {'loss': [], 'iou': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 30)
        
        # 训练
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], epoch
        )
        
        # 验证
        val_loss, val_iou = validate_epoch(
            model, val_loader, criterion, config['device'], epoch
        )
        
        # 记录历史
        train_history['loss'].append(train_loss)
        train_history['iou'].append(train_iou)
        train_history['val_loss'].append(val_loss)
        train_history['val_iou'].append(val_iou)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"保存最佳模型，验证集IoU: {val_iou:.4f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f"训练损失: {train_loss:.4f}, 训练IoU: {train_iou:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}")
    
    # ========== 训练结果可视化 ==========
    print("\n" + "="*50)
    print("训练完成！绘制训练曲线...")
    
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='训练损失')
    plt.plot(train_history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # IoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_history['iou'], label='训练IoU')
    plt.plot(train_history['val_iou'], label='验证IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('训练和验证IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.show()
    
    print(f"最佳验证集IoU: {best_val_iou:.4f}")
    print(f"模型和训练曲线已保存到: {config['save_dir']}")


if __name__ == "__main__":
    # 训练模型
    main()
    