# Unet 图像分割
使用数据集 PlantVillage，该数据集包含 color、grayscale、segmented 三种图像，Unet 分割使用到其中的 color、segmented，color 文件夹中的图像相当于数据，segmented 文件夹中的图像相当于标签，按照 [数据划分脚本](./split_image.py) 将图像以 8:1:1 的比例划分为训练、验证、测试。  

编码器部分采用 mobilenet-v2，其权重采用 torchvision 中预训练的 mobilenet-v2 权重，必要时可以冻结编码器的参数