import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import os
import random

# 设置随机种子（确保每次运行有一致性，可选）
# random.seed(42)
# torch.manual_seed(42)

def AddNoise():
    """自定义添加高斯噪声的 transform"""
    def apply_noise(img):
        tensor = F.to_tensor(img)
        noise = torch.randn_like(tensor) * 0.1  # 噪声强度
        noisy = tensor + noise
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return F.to_pil_image(noisy)
    return apply_noise

def create_augmentation_pipeline():
    """创建一系列增强操作的 pipeline"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),           # 50% 概率翻转
        transforms.RandomResizedCrop(480, scale=(0.8, 1.0)),  # 裁剪并调整到 224x224
        AddNoise(),                                       # 添加噪声
        transforms.RandomGrayscale(p=1.0),               # 20% 概率转灰度
        # 注意：不加 ToTensor 或 Normalize，因为我们想保存为PIL图像
    ])

# =============================
# 主函数：加载 → 增强 → 保存
# =============================
def augment_and_save(image_path, output_path):
    # 加载图像
    img = Image.open(image_path).convert("RGB")

    # 创建增强流水线
    transform = create_augmentation_pipeline()

    # 应用所有增强（注意：AddNoise 是自定义函数，也能被 Compose 支持）
    augmented_img = transform(img)

    # 保存最终结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    augmented_img.save(output_path)
    print(f"✅ 最终增强图像已保存至: {output_path}")

# === 使用示例 ===
if __name__ == "__main__":
    input_image = "assets/demo1.jpg"           # ← 替换为你的输入图片路径
    output_image = "assets/demo1_augmented.jpg"  # 输出路径

    augment_and_save(input_image, output_image)