import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import create_mobilenet_unet

def predict_single_image(model_path, image_path, output_path=None, device='cuda'):
    """对单张图像进行预测"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = create_mobilenet_unet(num_classes=config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 二值化
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # 调整回原始大小
    pred_mask_resized = cv2.resize(pred_mask_binary, 
                                  (original_size[0], original_size[1]),
                                  interpolation=cv2.INTER_NEAREST)
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, pred_mask_resized)
        print(f"预测结果已保存到: {output_path}")
    
    return pred_mask_resized

if __name__ == "__main__":
    # 原图
    origin_image=cv2.imread('./Apple_scab_3417.JPG')
    origin_image=cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    
    # 原图对应的标签，即分割图
    mask_image=cv2.imread('./Apple_scab_3417_final_masked.jpg')
    mask_image=cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    # 使用训练好的模型进行预测，得到预测出的掩模
    pred_mask=predict_single_image(
        model_path='checkpoints/best_model.pth',
        image_path='Apple_scab_3417.JPG',
        output_path='prediction.png'
    )

    # 创建 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(origin_image)
    axes[0].set_title('origin_image')
    axes[0].axis('off')

    axes[1].imshow(mask_image)
    axes[1].set_title('mask_image')
    axes[1].axis('off')

    axes[2].imshow(pred_mask,cmap='gray')
    axes[2].set_title('pred_mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    