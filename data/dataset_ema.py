from data.common import AbstractDataset
from torch.utils.data._utils.collate import default_collate
import torch
import data.transforms as custom_T
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将 normalized tensor 反归一化到 [0, 255]
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return (tensor * std + mean) * 255


def validate_inverse_matrix(aug_img, inv_matrix, orig_img, device='cpu'):
    """
    使用 inv_matrix 从 aug_img 重建 orig_img，并与真实 orig_img 对比

    Args:
        aug_img: 增强后的图像 (numpy array, HWC or CHW)
        inv_matrix: (3,3) numpy array, aug_space → orig_space
        orig_img: 原始图像 (numpy array, HWC or CHW)
        device: 'cpu' or 'cuda'

    Returns:
        reconstructed (CHW), valid_mask (HW), metrics
    """
    # 确保输入是 numpy array，并转为 CHW 格式
    if isinstance(aug_img, np.ndarray):
        if aug_img.ndim == 3 and aug_img.shape[0] == 3:  # CHW
            aug_np = aug_img  # already CHW
        elif aug_img.ndim == 3 and aug_img.shape[2] == 3:  # HWC
            aug_np = np.transpose(aug_img, (2, 0, 1))  # HWC -> CHW
        else:
            raise ValueError(f"Invalid aug_img shape: {aug_img.shape}")
    else:
        raise TypeError("aug_img must be numpy array")

    if isinstance(orig_img, np.ndarray):
        if orig_img.ndim == 3 and orig_img.shape[0] == 3:
            orig_np = orig_img
        elif orig_img.ndim == 3 and orig_img.shape[2] == 3:
            orig_np = np.transpose(orig_img, (2, 0, 1))
        else:
            raise ValueError(f"Invalid orig_img shape: {orig_img.shape}")
    else:
        raise TypeError("orig_img must be numpy array")

    # 转为 tensor
    aug_tensor = torch.tensor(aug_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, C, H, W)
    orig_tensor = torch.tensor(orig_np, dtype=torch.float32).to(device)  # (C, H, W)

    H_orig, W_orig = orig_tensor.shape[1:]
    H_aug, W_aug = aug_tensor.shape[2:]

    # 生成原始空间网格
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H_orig, device=device),
        torch.linspace(-1, 1, W_orig, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H_orig, W_orig, 2)
    grid_flat = grid.reshape(-1, 2)  # (N, 2)

    # 转为原始像素坐标
    orig_pixel = grid_flat * torch.tensor([[W_orig, H_orig]], device=device) / 2 + \
                 torch.tensor([[W_orig, H_orig]], device=device) / 2  # (N, 2)

    # 齐次坐标
    orig_pixel_hom = torch.cat([
        orig_pixel,
        torch.ones(len(orig_pixel), 1, device=device)
    ], dim=1)  # (N, 3)

    # 映射到增强图像空间
    inv_matrix = torch.tensor(inv_matrix, dtype=torch.float32, device=device)
    forward_matrix = torch.inverse(inv_matrix)
    aug_pixel_hom = forward_matrix @ orig_pixel_hom.T  # (3, N)
    aug_pixel_hom = aug_pixel_hom.T  # (N, 3)
    aug_pixel = aug_pixel_hom[:, :2] / (aug_pixel_hom[:, 2:] + 1e-8)  # (N, 2)

    # valid mask
    valid_x = (aug_pixel[:, 0] >= 0) & (aug_pixel[:, 0] < W_aug)
    valid_y = (aug_pixel[:, 1] >= 0) & (aug_pixel[:, 1] < H_aug)
    valid_mask_1d = valid_x & valid_y
    valid_mask = valid_mask_1d.reshape(H_orig, W_orig)  # (H, W)

    # 归一化坐标 for grid_sample
    aug_norm = (aug_pixel - torch.tensor([[W_aug, H_aug]], device=device) / 2) * \
               2 / torch.tensor([[W_aug, H_aug]], device=device)
    sampling_grid = aug_norm.reshape(1, H_orig, W_orig, 2)

    # 采样
    reconstructed = F.grid_sample(
        aug_tensor,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).squeeze(0)  # (C, H, W)

    # -------------------------------
    # 处理是否归一化（关键！）
    # 假设 aug_img 是经过 Normalize 的（如 student transform），则需反归一化
    # 否则（如仅 ToTensor）则不需要
    # -------------------------------

    # 注意：reconstructed 是采样结果，如果 aug_tensor 是归一化的，则 reconstructed 也是
    # 我们需要判断是否要反归一化
    # 策略：如果 aug_tensor.mean() < 1，则大概率是 [0,1]；否则可能是归一化后的
    # if aug_tensor.max() > 1.5:  # 可能是 [0,255]
    #     recon_pixel = reconstructed
    # elif aug_tensor.max() <= 1.5 and aug_tensor.std() < 0.5:  # 很可能是归一化后的
    recon_pixel = denormalize(reconstructed.unsqueeze(0)).squeeze(0)  # to [0,255]
    # else:
    #     recon_pixel = reconstructed * 255  # to [0,255]

    orig_pixel = orig_tensor * 255 if orig_tensor.max() <= 1 else orig_tensor

    # masked MSE
    mse_loss = F.mse_loss(recon_pixel, orig_pixel, reduction='none')  # (C, H, W)
    masked_mse = (mse_loss * valid_mask).sum() / valid_mask.sum()
    psnr = 10 * torch.log10((255 ** 2) / masked_mse)

    metrics = {
        'masked_mse': masked_mse.item(),
        'psnr': psnr.item(),
        'valid_ratio': valid_mask.float().mean().item()
    }

    return reconstructed.cpu(), valid_mask.cpu(), metrics

class StudentTeacherDataset(AbstractDataset):
    
    @staticmethod
    def get_default_student_transforms():
        transforms = custom_T.Compose(
            [
                custom_T.Resize(480), 
                custom_T.RandomColorJitter(0.1, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                custom_T.RandomGaussianBlur(0.5, kernel_size=11, sigma_min=0.1, sigma_max=2.0),
                custom_T.RandomNoise(0.5, noise_level=0.2),
                custom_T.RandomCrop(1.0, 360, 480),
                custom_T.Resize(480),
                custom_T.RandomHorizontalFlip(0.5),
                custom_T.RandomGrayScale(0.3),
                custom_T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                custom_T.ToTensor()
            ]
        )
        return transforms

    @staticmethod
    def get_default_teacher_transforms():
        transforms = custom_T.Compose(
            [
                custom_T.Resize(480),
                custom_T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                custom_T.ToTensor()
            ]
        )
        return transforms
    
    def __init__(self, root: str = "/data/datasets/tzhangbu/Cherry-Pick/data/refcoco", 
                 dataset: str = "unc",
                 split = "train", 
                 max_tokens=20,
                 max_iters = None,
                 teacher_transforms=None,
                 student_transforms=None,
                 ):
        super().__init__(
            root=root, 
            dataset=dataset, 
            split=split, 
            max_tokens=max_tokens, 
            image_transforms=None
        )
        self.max_iters = max_iters
        self.teacher_transforms = teacher_transforms if teacher_transforms is not None else self.get_default_teacher_transforms()
        self.student_transforms = student_transforms if student_transforms is not None else self.get_default_student_transforms()
        
        self.index_files = sorted(os.listdir(self.index_root), key=self.extract_number)
        if self.max_iters is not None:
            self.index_files = self.index_files[:self.max_iters]
    
    def __len__(self):
        return len(self.index_files)

    def __getitem__(self, idx):
        index_path = os.path.join(self.index_root, self.index_files[idx])
        img, mask_array, txt, similarity_scores, predicted_mask_id = self.load_from_index(index_path)
        normalized_scores = self.normalize_to_softmax(similarity_scores)
        target_loss_weight = normalized_scores[predicted_mask_id]
        
        input_teacher = {
            "img": img,
            "mask": mask_array,
            "text": txt
        }
        
        input_student = {
            "img": img,
            "mask": mask_array,
            "text": txt
        }
        
        input_teacher, inv_teacher = self.teacher_transforms(input_teacher)
        input_student, inv_student = self.student_transforms(input_student)  
        
        teacher_input_ids, teacher_attention_mask = self.tokenize_text(input_teacher["text"])
        student_input_ids, student_attention_mask = self.tokenize_text(input_student["text"])
        
        return {
            "teacher": {
                "image": input_teacher["img"],
                "mask": input_teacher["mask"],
                "text": input_teacher["text"],
                "input_ids": teacher_input_ids,
                "attention_mask": teacher_attention_mask,
                "inv": torch.from_numpy(inv_teacher).float()
            },
            "student": {
                "image": input_student["img"],
                "mask": input_student["mask"],
                "text": input_student["text"],
                "input_ids": student_input_ids,
                "attention_mask": student_attention_mask,
                "inv": torch.from_numpy(inv_student).float()
            },
            "orig_img": img,
            "sup_loss_weight": target_loss_weight,
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: list of outputs from __getitem__
        
        Returns:
            collated batch with:
                - 'orig_img' as list of np.ndarray
                - others collated by default_collate (stacked, tensorized)
        """
        if len(batch) == 0:
            return {}
        orig_imgs = [item.pop("orig_img") for item in batch]  # 提取并移除 orig_img
        collated = default_collate(batch)
        collated["orig_img"] = orig_imgs  # list of np.ndarray
        return collated
    

if __name__ == "__main__":
    dataset = StudentTeacherDataset(
        root="/localdata/tzhangbu/dataset/refcoco",
        dataset="unc",
        split="train",
        max_iters=50
    )

    # for i in range(len(dataset)):
    #     sample = dataset[i]

    #     orig_img = sample["orig_img"]           # numpy array (HWC)
    #     teacher_img = sample["teacher"]["image"].cpu().numpy()  # (3, H, W)
    #     student_img = sample["student"]["image"].cpu().numpy()  # (3, H, W)
    #     teacher_inv = sample["teacher"]["inv"].cpu().numpy()    # (3,3)
    #     student_inv = sample["student"]["inv"].cpu().numpy()    # (3,3)

    #     print(f"\n--- Sample {i} ---")
    #     print("Teacher inv matrix:\n", teacher_inv)
    #     print(f"Original shape: {orig_img.shape}")

    #     # 🔹 验证 teacher_inv
    #     print("1. Validating teacher_inv")
    #     rec_t, mask_t, met_t = validate_inverse_matrix(
    #         teacher_img,      # numpy CHW
    #         teacher_inv,
    #         orig_img,         # numpy HWC or CHW → 自动处理
    #         device='cpu'
    #     )
    #     print(f"   Masked MSE: {met_t['masked_mse']:.4f}, PSNR: {met_t['psnr']:.2f} dB, Valid Ratio: {met_t['valid_ratio']:.2f}")

    #     # 🔹 验证 student_inv
    #     print("2. Validating student_inv")
    #     rec_s, mask_s, met_s = validate_inverse_matrix(
    #         student_img,
    #         student_inv,
    #         orig_img,
    #         device='cpu'
    #     )
    #     print(f"   Masked MSE: {met_s['masked_mse']:.4f}, PSNR: {met_s['psnr']:.2f} dB, Valid Ratio: {met_s['valid_ratio']:.2f}")

    #     # 🔹 可视化（纯 numpy）
    #     rec_t_img = rec_t.numpy()  # (3, H, W)
    #     rec_s_img = rec_s.numpy()
    #     orig_img_chw = orig_img if orig_img.shape[0] == 3 else np.transpose(orig_img, (2, 0, 1))
    #     orig_img_chw = orig_img_chw.astype(np.float32)

    #     # 反归一化 reconstructed（因为它们是模型输出，可能是归一化后的）
    #     # 这里我们假设 rec_t 和 rec_s 是归一化后的，所以反归一化
    #     mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    #     std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    #     rec_t_denorm = np.clip((rec_t_img * std + mean) * 255, 0, 255).astype(np.uint8)
    #     rec_s_denorm = np.clip((rec_s_img * std + mean) * 255, 0, 255).astype(np.uint8)
    #     orig_uint8 = np.clip(orig_img_chw * 255, 0, 255).astype(np.uint8) if orig_img_chw.max() <= 1 \
    #                  else orig_img_chw.astype(np.uint8)

    #     # 转为 HWC 显示
    #     rec_t_hwc = np.transpose(rec_t_denorm, (1, 2, 0))
    #     rec_s_hwc = np.transpose(rec_s_denorm, (1, 2, 0))
    #     orig_hwc = np.transpose(orig_uint8, (1, 2, 0))

    #     # 差值图
    #     diff_t = np.abs(rec_t_hwc.astype(np.int32) - orig_hwc.astype(np.int32)).astype(np.uint8)
    #     diff_s = np.abs(rec_s_hwc.astype(np.int32) - orig_hwc.astype(np.int32)).astype(np.uint8)
    #     mask_s_np = mask_s.numpy()
    #     diff_s_masked = diff_s * np.stack([mask_s_np]*3, axis=-1)

    #     # 绘图
    #     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    #     axes[0,0].imshow(rec_t_hwc)
    #     axes[0,0].set_title("Reconstructed (Teacher)")
    #     axes[0,1].imshow(rec_s_hwc)
    #     axes[0,1].set_title("Reconstructed (Student)")
    #     axes[0,2].imshow(orig_hwc)
    #     axes[0,2].set_title("Original (Ground Truth)")
    #     axes[0,3].imshow(mask_s_np, cmap='gray')
    #     axes[0,3].set_title("Valid Mask (Student)")

    #     axes[1,0].imshow(diff_t)
    #     axes[1,0].set_title("Diff (Teacher)")
    #     axes[1,1].imshow(diff_s)
    #     axes[1,1].set_title("Diff (Student)")
    #     axes[1,2].imshow(diff_s_masked)
    #     axes[1,2].set_title("Diff (Student, Masked)")
    #     axes[1,3].axis('off')

    #     for ax in axes.flat:
    #         ax.axis('off')
    #     plt.tight_layout()
    #     # plt.savefig(f"validation_sample_{i}.png")
    #     # plt.close()
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=StudentTeacherDataset.collate_fn
    )
    
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Teacher image shape:", batch["teacher"]["image"].shape)
        print("Student image shape:", batch["student"]["image"].shape)
        print("Orig img list length:", len(batch["orig_img"]))
    
        
        
        
        
    
    


    
    

    
        
        