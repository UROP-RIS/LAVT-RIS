import copy
import cv2
import numpy as np
import random
import re
import torch
import torch.nn.functional as F


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        h, w = img.shape[:2]

        if random.random() < self.p:
            out_dict['img'] = img[:, ::-1, ...]
            out_dict['mask'] = out_dict['mask'][:, ::-1, ...]
            
            text = out_dict['text']
            text = re.sub(r'\bright\b', '###TEMP###', text)
            text = re.sub(r'\bleft\b', 'right', text)
            text = re.sub(r'###TEMP###', 'left', text)
            out_dict['text'] = text
            
            text = out_dict["aug_text"]
            text = re.sub(r'\bright\b', '###TEMP###', text)
            text = re.sub(r'\bleft\b', 'right', text)
            text = re.sub(r'###TEMP###', 'left', text)
            out_dict['aug_text'] = text

            inv_matrix = np.array([
                [-1,  0, w - 1],
                [ 0,  1,     0],
                [ 0,  0,     1]
            ])
        else:
            inv_matrix = np.eye(3)

        return out_dict, inv_matrix

class RandomMaskImage:
    
    def __init__(self, p=0.5, patch_size=16, mask_ratio=0.75):
        self.p = p
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']  # 假设 img 是 H x W x C 的 numpy 数组，且 H, W 可被 patch_size 整除

        # 随机决定是否应用 mask
        if random.random() > self.p:
            inverse_matrix = np.eye(3, dtype=np.float32)
            return out_dict, inverse_matrix

        h, w = img.shape[:2]
        ph = pw = self.patch_size

        # 断言：图像尺寸必须能被 patch_size 整除
        assert h % ph == 0, f"Image height {h} is not divisible by patch_size {ph}"
        assert w % pw == 0, f"Image width {w} is not divisible by patch_size {pw}"

        gh, gw = h // ph, w // pw  # 网格数量
        num_patches = gh * gw
        num_masked = int(num_patches * self.mask_ratio)
        indices = np.random.permutation(num_patches)
        masked_indices = indices[:num_masked]
        mask = np.zeros((gh, gw), dtype=bool)
        mask.flat[masked_indices] = True
        img_masked = img.copy()
        for i in range(gh):
            for j in range(gw):
                if mask[i, j]:
                    start_h, end_h = i * ph, (i + 1) * ph
                    start_w, end_w = j * pw, (j + 1) * pw
                    img_masked[start_h:end_h, start_w:end_w, :] = 0  # 或使用均值等填充

        out_dict['img'] = img_masked
        out_dict['mask'] = mask                    # 可用于重建
        out_dict['masked_indices'] = masked_indices
        out_dict['unmask_indices'] = np.logical_not(mask).flat.nonzero()[0]
        inverse_matrix = np.eye(3, dtype=np.float32)

        return out_dict, inverse_matrix

class RandomMaskText:
    
    def __init__(self, p=0.5, mask_ratio=0.15, mask_token='[MASK]'):
        self.p = p
        self.mask_token = mask_token
        self.mask_ratio = mask_ratio

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        text = out_dict['text']

        if random.random() > self.p:
            inverse_matrix = np.eye(3, dtype=np.float32)
            return out_dict, inverse_matrix

        words = re.findall(r'\S+', text)
        if len(words) <= 1:
            inverse_matrix = np.eye(3, dtype=np.float32)
            return out_dict, inverse_matrix

        num_words = len(words)
        num_masked = int(num_words * self.mask_ratio)
        num_masked = max(1, min(num_masked, num_words))  
        indices = np.random.choice(num_words, num_masked, replace=False)

        masked_words = words.copy()
        for idx in indices:
            masked_words[idx] = self.mask_token
            
        masked_text = ' '.join(masked_words)
        masked_text = re.sub(r'(?:\s*\[MASK\]\s*)+', f' {self.mask_token} ', masked_text)
        masked_text = re.sub(r'\s+', ' ', masked_text).strip()
        out_dict['text'] = masked_text
        inverse_matrix = np.eye(3, dtype=np.float32)

        return out_dict, inverse_matrix

class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        h, w = img.shape[:2]

        scale_h = self.size / h
        scale_w = self.size / w

        new_w, new_h = self.size, self.size
        # Resize
        out_dict['img'] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out_dict['mask'] = cv2.resize(
            out_dict['mask'], (new_w, new_h), 
            interpolation=cv2.INTER_NEAREST
        )

        # 逆变换矩阵：aug → orig
        inv_matrix = np.array([
            [1/scale_w,       0, 0],
            [      0, 1/scale_h, 0],
            [      0,       0, 1]
        ])

        return out_dict, inv_matrix
        
    
class RandomResize:
    def __init__(self, p, sizes, resize_long=True):
        self.sizes = [sizes] if isinstance(sizes, (int, float)) else sizes
        self.resize_long = resize_long
        self.p = p

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        if random.random() > self.p:
            inverse_matrix = np.eye(3)
            return out_dict, inverse_matrix
        h, w = img.shape[:2]

        size = random.choice(self.sizes)
        if self.resize_long:
            scale = size / max(w, h)
        else:
            scale = size / min(w, h)

        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        out_dict['img'] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out_dict['mask'] = cv2.resize(
            out_dict['mask'], (new_w, new_h), 
            interpolation=cv2.INTER_NEAREST
        )

        # 逆变换矩阵：aug → orig
        inv_matrix = np.array([
            [1/scale,       0, 0],
            [      0, 1/scale, 0],
            [      0,       0, 1]
        ])

        return out_dict, inv_matrix
    
class RandomCrop:
    def __init__(self, p, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.p = p

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        if random.random() > self.p:
            inverse_matrix = np.eye(3)
            return out_dict, inverse_matrix
        h, w = img.shape[:2]

        crop_h = random.randint(self.min_size, min(h, self.max_size))
        crop_w = random.randint(self.min_size, min(w, self.max_size))

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # Crop
        out_dict['img'] = img[top:top+crop_h, left:left+crop_w]
        out_dict['mask'] = out_dict['mask'][top:top+crop_h, left:left+crop_w]

        # 逆变换矩阵：aug → orig
        inv_matrix = np.array([
            [1, 0, left],
            [0, 1,  top],
            [0, 0,    1]
        ])

        return out_dict, inv_matrix

class RandomColorJitter:
    def __init__(self, p, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        """
        Args:
            brightness (float): 亮度抖动幅度，范围 [0, 1]
            contrast (float):   对比度抖动幅度，范围 [0, 1]
            saturation (float): 饱和度抖动幅度，范围 [0, 1]
            hue (float):        色相抖动幅度，范围 [0, 0.5]
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, input_dict):
        # 深拷贝
        out_dict = copy.deepcopy(input_dict)

        img = out_dict['img']  # (H, W, 3), uint8, HWC
        
        if random.random() > self.p:
            inverse_matrix = np.eye(3)
            return out_dict, inverse_matrix

        # 转为 float32 [0, 1]
        img = img.astype(np.float32) / 255.0

        # 随机顺序应用四种抖动（DETR / SimCLR 风格）
        # 注意：hue 和 saturation 最好在 HSV 空间处理
        transform_order = list(range(4))
        random.shuffle(transform_order)

        for op_id in transform_order:
            if op_id == 0 and self.brightness > 0:
                # Brightness: 在 Luma 空间加偏移
                delta = random.uniform(-self.brightness, self.brightness)
                img += delta
                img = np.clip(img, 0, 1)

            elif op_id == 1 and self.contrast > 0:
                # Contrast: 缩放像素值
                alpha = random.uniform(1 - self.contrast, 1 + self.contrast)
                img = (img - 0.5) * alpha + 0.5
                img = np.clip(img, 0, 1)

            elif op_id == 2 and self.saturation > 0:
                # Saturation: 转换到 HSV，调整 S
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                alpha = random.uniform(1 - self.saturation, 1 + self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 1)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                img = np.clip(img, 0, 1)

            elif op_id == 3 and self.hue > 0:
                # Hue: 在 HSV 的 H 通道加偏移（需 wrap around）
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                delta = random.uniform(-self.hue, self.hue)
                hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 1.0  # hue 是 [0,1]
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                img = np.clip(img, 0, 1)

        out_dict['img'] = (img * 255).astype(np.uint8)
        inverse_matrix = np.eye(3)

        return out_dict, inverse_matrix

class RandomGaussianBlur:
    
    def __init__(self, p, kernel_size=5, sigma_min=0.1, sigma_max=2.0):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        if random.random() > self.p:
            inverse_matrix = np.eye(3)
            return out_dict, inverse_matrix

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)
        out_dict['img'] = img
        inverse_matrix = np.eye(3)
        return out_dict, inverse_matrix

class RandomGrayScale:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        if random.random() > self.p:
            inverse_matrix = np.eye(3)
            return out_dict, inverse_matrix

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_3c = np.stack([gray, gray, gray], axis=-1)
        out_dict['img'] = gray_3c
        inverse_matrix = np.eye(3)
        return out_dict, inverse_matrix

class RandomNoise:
    def __init__(self, p=0.1, noise_level=0.05):
        self.p = p
        self.noise_level = noise_level

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        if random.random() > self.p:
            inverse_matrix = np.eye(3)
            return out_dict, inverse_matrix

        noise = np.random.randn(*img.shape) * 255 * self.noise_level
        noisy_img = img.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        out_dict['img'] = noisy_img
        inverse_matrix = np.eye(3)
        return out_dict, inverse_matrix

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img'].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        out_dict['img'] = img
        inverse_matrix = np.eye(3)
        return out_dict, inverse_matrix
    
class ToTensor:
    def __call__(self, input_dict):
        out_dict = copy.deepcopy(input_dict)
        img = out_dict['img']
        mask = out_dict['mask']
        # HWC -> CHW
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask)
        out_dict['img'] = img
        out_dict['mask'] = mask
        
        inverse_matrix = np.eye(3)
        return out_dict, inverse_matrix        
        


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms  # list of augmentations

    def __call__(self, input_dict):
        """
        顺序应用所有变换
        返回: (final_input_dict, total_inverse_matrix)
        """
        total_inv_matrix = np.eye(3)
        data = input_dict

        for t in self.transforms:
            data, inv_matrix = t(data)
            total_inv_matrix = total_inv_matrix @ inv_matrix  # 累积：T_total = T3 @ T2 @ T1

        return data, total_inv_matrix

def cross_align_features(
    teacher_feat: torch.Tensor,
    teacher_inv: torch.Tensor,
    student_feat: torch.Tensor,
    student_inv: torch.Tensor,
    mode: str = 'bilinear',
    align_corners: bool = True,
):
    """
    Align teacher features (in teacher-augmented space) to student-augmented space.

    Args:
        teacher_feat: (B, C, H_t, W_t), features in teacher's augmented view
        teacher_inv: (B, 3, 3), matrix: teacher_space → orig_space
        student_feat: (B, C, H_s, W_s), used for output size
        student_inv: (B, 3, 3), matrix: student_space → orig_space
        mode: interpolation mode ('bilinear', 'nearest')
        align_corners: whether to use align_corners in grid_sample

    Returns:
        aligned_teacher: (B, C, H_s, W_s), aligned to student's view
    """
    B, C, H_s, W_s = student_feat.shape
    device = teacher_feat.device

    # -------------------------------
    # Step 1: 生成 student 空间的目标网格 [-1,1]
    # -------------------------------
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H_s, device=device),
        torch.linspace(-1, 1, W_s, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H_s, W_s, 2)
    grid_flat = grid.reshape(-1, 2)  # (N, 2)
    grid_flat = grid_flat.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

    # -------------------------------
    # Step 2: 转为 student 像素坐标 → 齐次坐标
    # -------------------------------
    student_pixel = grid_flat * torch.tensor([[[W_s, H_s]]], device=device) / 2 + \
                    torch.tensor([[[W_s, H_s]]], device=device) / 2  # (B, N, 2)
    
    student_pixel_hom = torch.cat([
        student_pixel,
        torch.ones(B, student_pixel.shape[1], 1, device=device)
    ], dim=-1)  # (B, N, 3)

    # -------------------------------
    # Step 3: student_space → orig_space (使用 student_inv)
    # -------------------------------
    orig_pixel_hom = torch.bmm(student_inv, student_pixel_hom.transpose(1, 2))  # (B, 3, N)
    orig_pixel_hom = orig_pixel_hom.transpose(1, 2)  # (B, N, 3)
    orig_pixel = orig_pixel_hom[:, :, :2] / (orig_pixel_hom[:, :, 2:] + 1e-8)  # (B, N, 2)

    # -------------------------------
    # Step 4: orig_space → teacher_space (使用 teacher_inv 的逆)
    # -------------------------------
    try:
        teacher_forward = torch.inverse(teacher_inv)  # (B, 3, 3): orig → teacher
    except:
        teacher_forward = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        print("Warning: teacher_inv not invertible!")

    teacher_pixel_hom = torch.bmm(teacher_forward, 
                                  torch.cat([orig_pixel, torch.ones_like(orig_pixel[:, :, :1])], dim=-1).transpose(1, 2))
    teacher_pixel_hom = teacher_pixel_hom.transpose(1, 2)  # (B, N, 3)
    teacher_pixel = teacher_pixel_hom[:, :, :2] / (teacher_pixel_hom[:, :, 2:] + 1e-8)  # (B, N, 2)

    # -------------------------------
    # Step 5: 转为归一化坐标 [-1,1] for grid_sample
    # -------------------------------
    H_t, W_t = teacher_feat.shape[2:]
    teacher_norm = (teacher_pixel - torch.tensor([[[W_t, H_t]]], device=device) / 2) * \
                   2 / torch.tensor([[[W_t, H_t]]], device=device)  # (B, N, 2)
    sampling_grid = teacher_norm.reshape(B, H_s, W_s, 2)

    # -------------------------------
    # Step 6: 从 teacher_feat 中采样
    # -------------------------------
    aligned_teacher = F.grid_sample(
        teacher_feat,
        sampling_grid,
        mode=mode,
        padding_mode='border',
        align_corners=align_corners
    )  # (B, C, H_s, W_s)

    return aligned_teacher



if __name__ == "__main__":
        # 模拟输入
    B, C, H_s, W_s = 4, 256, 48, 48
    H_t, W_t = 64, 64

    student_feat = torch.randn(B, C, H_s, W_s)

    # 每个样本有一个逆变换矩阵（例如来自 RandomResize + RandomCrop）
    inv_matrices_stu = []
    inv_matrices_teacher = []
    for _ in range(B):
        # 示例：缩放 + 平移
        scale = 0.8 + 0.4 * torch.rand(1).item()  # 0.8 ~ 1.2
        tx, ty = torch.rand(2).tolist()
        T = torch.tensor([
            [scale, 0, tx * 100],
            [0, scale, ty * 100],
            [0, 0, 1]
        ])
        inv_matrices_stu.append(torch.inverse(T))
        inv_matrices_teacher.append(np.eye(3))  # 假设 teacher 没有变换
    inv_matrix_tensor = torch.stack(inv_matrices_stu)  # (B, 3, 3)

    # 对齐
    aligned = cross_align_features(
        teacher_feat=torch.randn(B, C, H_t, W_t),
        teacher_inv=inv_matrix_tensor,
        student_feat=student_feat,
        student_inv=inv_matrix_tensor,
        mode='bilinear',
        align_corners=True
    )
    print(aligned.shape)

    horizontal_transform = RandomHorizontalFlip(p=1.0)
    input_dict = {
        'img': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'mask': np.random.randint(0, 10, (480, 640), dtype=np.uint8),
        'text': 'A person on the left and a car on the right.'
    }
    out_dict, inv_matrix = horizontal_transform(input_dict)
    print("Original text:", input_dict['text'])
    print("Flipped text:", out_dict['text'])
    
    
    
    
