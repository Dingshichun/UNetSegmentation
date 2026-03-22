import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

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

    # 使用训练好的模型进行预测
    predict_single_image(
        model_path='checkpoints/best_model.pth',
        image_path='Apple_scab_3417.JPG',
        output_path='prediction.png'
    )