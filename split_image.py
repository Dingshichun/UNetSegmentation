import os
import shutil
import random
import numpy as np
from pathlib import Path

def prepare_plantvillage_dataset_nested(color_dir, segmented_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    准备PlantVillage数据集用于分割训练（支持子文件夹结构）
    
    参数：
        color_dir: color文件夹路径，包含38个子文件夹
        segmented_dir: segmented文件夹路径，包含相同的38个子文件夹结构
        output_dir: 输出目录
        split_ratio: (训练集比例, 验证集比例, 测试集比例)
    """
    
    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    
    # 获取color文件夹下的所有子文件夹（38个类别）
    categories = [d for d in os.listdir(color_dir) 
                  if os.path.isdir(os.path.join(color_dir, d))]
    
    print(f"发现 {len(categories)} 个类别文件夹")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    # 设置随机种子确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 对每个类别单独处理
    for category in categories:
        print(f"\n处理类别: {category}")
        
        # 构建路径
        color_category_dir = os.path.join(color_dir, category)
        segmented_category_dir = os.path.join(segmented_dir, category)
        
        # 检查segmented文件夹是否存在对应的类别文件夹
        if not os.path.exists(segmented_category_dir):
            print(f"警告: segmented文件夹中没有找到类别 '{category}'，跳过")
            continue
        
        # 获取该类别下的所有图像文件
        color_files = []
        for ext in ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG']:
            color_files.extend([f for f in os.listdir(color_category_dir) if f.lower().endswith(ext)])
        
        if not color_files:
            print(f"警告: 类别 '{category}' 中没有找到图像文件，跳过")
            continue
        
        # 移除可能的重复项（按文件名，不区分大小写）
        color_files = list(set(color_files))
        color_files.sort()  # 排序以确保可重复性
        
        print(f"  找到 {len(color_files)} 张图像")
        
        # 随机打乱
        random.shuffle(color_files)
        
        # 计算分割点
        total = len(color_files)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        # 分割文件
        train_files = color_files[:train_end]
        val_files = color_files[train_end:val_end]
        test_files = color_files[val_end:]
        
        # 复制函数
        def copy_category_files(files, split_name, category_name):
            copied_count = 0
            for file in files:
                # 构建图像源路径
                img_src = os.path.join(color_category_dir, file)
                
                # 构建图像目标路径（添加类别前缀以避免文件名冲突）
                img_dst_name = f"{category_name}_{file}"
                img_dst = os.path.join(output_dir, split_name, 'images', img_dst_name)
                
                # 构建掩模源路径（尝试不同的扩展名）
                base_name, ext = os.path.splitext(file)
                mask_found = False
                
                # 首先尝试相同文件名
                mask_src = os.path.join(segmented_category_dir, file)
                if os.path.exists(mask_src):
                    mask_found = True
                
                if not mask_found:
                    # 观察下载的图像可知，segmented 文件中的图像命名比 color 文件夹多“_final_masked”
                    # 所以加上再在 segmented 文件夹查找
                    for mask_ext in ['.png', '.jpg', '.PNG', '.JPG']:
                        mask_src = os.path.join(segmented_category_dir, base_name + "_final_masked" + mask_ext)
                        if os.path.exists(mask_src):
                            mask_found = True
                            break
                
                if not mask_found:
                    print(f"  警告: 找不到掩模文件对应图像: {file}")
                    continue
                
                # 构建掩模目标路径
                mask_dst_name = f"{category_name}_{os.path.basename(mask_src)}"
                mask_dst = os.path.join(output_dir, split_name, 'masks', mask_dst_name)
                
                # 复制文件
                try:
                    shutil.copy2(img_src, img_dst)
                    shutil.copy2(mask_src, mask_dst)
                    copied_count += 1
                except Exception as e:
                    print(f"  错误: 复制文件时出错 {file}: {e}")
            
            return copied_count
        
        # 复制各类别的文件
        train_copied = copy_category_files(train_files, 'train', category)
        val_copied = copy_category_files(val_files, 'val', category)
        test_copied = copy_category_files(test_files, 'test', category)
        
        total_train += train_copied
        total_val += val_copied
        total_test += test_copied
        
        print(f"  成功复制: 训练集 {train_copied} 张, 验证集 {val_copied} 张, 测试集 {test_copied} 张")
    
    print("\n" + "="*50)
    print("数据集准备完成！")
    print(f"总统计:")
    print(f"训练集: {total_train} 张图像")
    print(f"验证集: {total_val} 张图像")
    print(f"测试集: {total_test} 张图像")
    print(f"总计: {total_train + total_val + total_test} 张图像")
    
    # 保存数据集划分信息
    info_file = os.path.join(output_dir, "dataset_info.txt")
    with open(info_file, 'w') as f:
        f.write("PlantVillage数据集划分信息\n")
        f.write("="*50 + "\n")
        f.write(f"数据来源:\n")
        f.write(f"  color目录: {color_dir}\n")
        f.write(f"  segmented目录: {segmented_dir}\n")
        f.write(f"划分比例: {split_ratio}\n")
        f.write(f"类别数量: {len(categories)}\n")
        f.write(f"训练集图像数量: {total_train}\n")
        f.write(f"验证集图像数量: {total_val}\n")
        f.write(f"测试集图像数量: {total_test}\n")
        f.write(f"总图像数量: {total_train + total_val + total_test}\n")
        f.write(f"随机种子: 42\n")
    
    print(f"数据集信息已保存到: {info_file}")

def analyze_dataset_structure(dataset_path):
    """
    分析数据集结构，确保正确性
    """
    print("\n分析数据集结构...")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(dataset_path, split, 'images')
        masks_dir = os.path.join(dataset_path, split, 'masks')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"警告: {split} 集缺少 images 或 masks 目录")
            continue
        
        images = sorted(os.listdir(images_dir))
        masks = sorted(os.listdir(masks_dir))
        
        # 检查图像和掩模数量是否匹配
        if len(images) != len(masks):
            print(f"警告: {split} 集图像({len(images)})和掩模({len(masks)})数量不匹配!")
        
        # 检查文件名对应关系
        mismatched = 0
        for img, mask in zip(images[:min(5, len(images))], masks[:min(5, len(masks))]):
            img_base = img.rsplit('_', 1)[-1].rsplit('.', 1)[0]
            mask_base = mask.rsplit('_', 1)[-1].rsplit('.', 1)[0]
            
            # 简单检查：图像和掩模的基础名应该相同
            if img_base != mask_base:
                mismatched += 1
                if mismatched <= 3:  # 只显示前几个不匹配的
                    print(f"  不匹配: {img} <-> {mask}")
        
        print(f"{split}集: {len(images)} 张图像, {len(masks)} 个掩模")

if __name__ == "__main__":
    # 使用示例
    color_dir = "./PlantVillage/color"  # 包含38个子文件夹的color目录
    segmented_dir = "./PlantVillage/segmented"  # 包含相同子文件夹结构的segmented目录
    output_dir = "./PlantVillage_processed"
    
    # 准备数据集
    prepare_plantvillage_dataset_nested(
        color_dir=color_dir,
        segmented_dir=segmented_dir,
        output_dir=output_dir,
        split_ratio=(0.7, 0.2, 0.1)  # 训练集70%，验证集20%，测试集10%
    )
    
    # 分析数据集结构
    analyze_dataset_structure(output_dir)
    
    # 可选：检查样本
    print("\n检查前5个样本:")
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(output_dir, split, 'images')
        images = sorted(os.listdir(images_dir))[:5]
        if images:
            print(f"{split}集样本: {images}")
