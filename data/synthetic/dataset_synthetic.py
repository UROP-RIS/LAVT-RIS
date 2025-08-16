import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from bert.tokenization_bert import BertTokenizer
import os
import re
import json
from pycocotools import mask as pycocotools_mask
import torch
from abc import abstractmethod
import transforms as T
import cv2
import random

class SynthesisDataset:
    
    def __init__(self, 
                 prob: float, 
                 root: str, 
                 dataset: str,
                 split: str,
                 max_tokens: int = 20, 
                 load_raw_data: bool = False,
                 use_shorter_dict: bool = True,
                 **kwargs):
        self.prob = prob
        self.max_tokens = max_tokens
        self.load_raw_data = load_raw_data
        
        self.root = root
        self.dataset = dataset
        self.split = split
        self.use_shorter_dict = use_shorter_dict
        
        self.index_root = f"{self.root}/{self.dataset}/{self.split}_purified_mask_list.json"
        self.image_txt_gt_root = f"{self.root}/{self.dataset}/{self.split}_batch"
        self.pseudo_label_root = f"{self.root}/{self.dataset}/{self.split}_mask_newB_batch"
        self.noun_dict_path = f"{self.root}/{self.dataset}/{self.dataset}_noun/{self.dataset}_{self.split}_dict.npy" if not use_shorter_dict else f"{self.root}/{self.dataset}/{self.dataset}_noun/{self.dataset}_noun.json"
        
        assert os.path.exists(self.index_root), f"Index file {self.index_root} does not exist."
        assert os.path.exists(self.image_txt_gt_root), f"Image and text ground truth root {self.image_txt_gt_root} does not exist."
        assert os.path.exists(self.pseudo_label_root), f"Pseudo label root {self.pseudo_label_root} does not exist."
        assert os.path.exists(self.noun_dict_path), f"Noun dictionary {self.noun_dict_path} does not exist."
        
        with open(self.index_root, 'r') as f:
            self.index = json.load(f)
        
        if not self.use_shorter_dict:
            with open(self.noun_dict_path, 'rb') as f:
                self.noun_dict = np.load(f, allow_pickle=True).item()
        
        else:
            with open(self.noun_dict_path, 'r') as f:
                self.noun_dict = json.load(f)
                print("using shorter dict")

        ## Image and mask transforms
        transforms = [T.Resize(480, 480),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]
        self.transforms = T.Compose(transforms)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.index)
    
    @abstractmethod
    def __call__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the synthesis operation to the input data.
        
        Args:
            index_data (dict): Input data containing image, text, and ground truth.
        
        Returns:
            tuple: Transformed image and mask, tokenized text, and attention mask.
        """
        pass
    
    def apply_transforms(self, img: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply image and mask transformations.
        
        Args:
            img (Image.Image): Input image.
            mask (Image.Image): Input mask.
        
        Returns:
            tuple: Transformed image and mask as tensors.
        """
        img, mask = self.transforms(img, mask)
        return img, mask
    
    def extract_number(self, filename):
        match = re.search(r'_(\d+)\.json$', filename)
        return int(match.group(1)) if match else -1
    
    def load(self, idx: dict):
        index_data = self.index[idx]
        mask_file_name = index_data["mask_file_name"]
        img_txt_gt_file_name = index_data["img_txt_gt_file_name"]
        predicted_mask_id = index_data["predicted_mask_id"]
        
        mask_file_path = os.path.join(self.pseudo_label_root, mask_file_name)
        img_txt_gt_path = os.path.join(self.image_txt_gt_root, img_txt_gt_file_name)

        img_txt_gt = np.load(img_txt_gt_path, allow_pickle=True)
        data_dict = {key: img_txt_gt[key] for key in img_txt_gt}
        img = data_dict['im_batch']
        txt = data_dict['sent_batch'][0]
        mask_candidates = json.load(open(mask_file_path, 'r'))["annotation"]
        rle_mask = mask_candidates[predicted_mask_id]["rle"]
        mask = pycocotools_mask.decode(rle_mask)
        
        ## Load noun
        try:
            noun = self.noun_dict[txt]
        except KeyError:
            noun = txt
        # img: np.array; txt: str; mask: np.array; noun: str
        return {"img": img, "txt": txt, "mask": mask, "noun": noun}
    
    def add_padding(self, img: np.ndarray, target_aspect: float, pad_value: int = 128) -> np.ndarray:
        """
        对图像添加 padding，保持宽高比，支持单通道和三通道图像
        
        Args:
            img: 输入图像 (H, W) 或 (H, W, 3), dtype=np.uint8
            target_aspect: 目标宽高比 (width / height)
            pad_value: 填充值（用于背景填充）
        
        Returns:
            padded_img: (H_out, W_out, 3) 或 (H_out, W_out) 与输入通道一致
        """
        is_gray = (len(img.shape) == 2)
        if is_gray:
            h, w = img.shape
            img_hwc = np.stack([img] * 3, axis=-1)  # 转为 (H, W, 3) 方便处理
        else:
            h, w = img.shape[:2]
            img_hwc = img
        current_aspect = w / h
        if current_aspect < target_aspect:
            new_w = int(h * target_aspect)
            new_h = h
            left = (new_w - w) // 2
            right = new_w - w - left
            top = bottom = 0
        else:
            new_h = int(w / target_aspect)
            new_w = w
            top = (new_h - h) // 2
            bottom = new_h - h - top
            left = right = 0
        padded = np.full((new_h, new_w, 3), pad_value, dtype=img_hwc.dtype)
        padded[top:top+h, left:left+w] = img_hwc
        if is_gray:
            padded = padded[:, :, 0]  # (H, W)

        return padded
    
    def create_patched_background(self, target_h, target_w, patch_size_range=(64, 256), num_patches=25, blur_kernel_ratio=0.02, target_brightness=None):
        """
        创建一个由随机 patch 拼接并模糊融合的背景，支持亮度匹配，用于提升合成图像真实性。

        Args:
            target_h (int): 目标背景高度
            target_w (int): 目标背景宽度
            patch_size_range (tuple): 裁剪 patch 的最小和最大尺寸 (min_size, max_size)
            num_patches (int): 用于拼接的 patch 数量
            blur_kernel_ratio (float): 模糊核大小占 min(target_h, target_w) 的比例
            target_brightness (float or None): 指定目标亮度值（0-255），若为 None 则从 patch 自动计算

        Returns:
            bg (np.ndarray): (H, W, 3) uint8，合成背景图
            bg_brightness (float): 返回背景的平均亮度，便于后续 patch 调整
        """
        bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        patches = []

        # === Step 1: 随机采样并裁剪多个 patch ===
        for _ in range(num_patches):
            idx = np.random.randint(0, len(self.index))
            data = self.load(idx)
            img = data['img']
            h, w = img.shape[:2]

            patch_size = random.randint(patch_size_range[0], patch_size_range[1])
            if h <= patch_size or w <= patch_size:
                continue

            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

        # === Step 2: 随机打乱并贴到画布上 ===
        random.shuffle(patches)
        for patch in patches:
            ph, pw = patch.shape[:2]
            max_x = target_w - pw
            max_y = target_h - ph
            if max_x <= 0 or max_y <= 0:
                continue
            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)
            bg[y:y+ph, x:x+pw] = patch

        # === Step 3: 全局模糊融合 ===
        kernel_size = int(blur_kernel_ratio * min(target_h, target_w))
        kernel_size = max(31, kernel_size // 2 * 2 + 1)  # 奇数
        bg = cv2.blur(bg, (kernel_size, kernel_size))

        # === Step 4: 亮度匹配准备 —— 计算背景亮度 ===
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        bg_brightness = bg_gray.mean()

        # 如果指定了目标亮度，调整整个背景
        if target_brightness is not None:
            ratio = target_brightness / (bg_brightness + 1e-5)
            bg = np.clip(bg.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
            bg_brightness = target_brightness

        return bg, bg_brightness
    
    def paste(self, bg: np.ndarray, patch: np.ndarray, patch_mask: np.ndarray, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = patch.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        x1, y1 = np.clip(x, 0, bg_w - w), np.clip(y, 0, bg_h - h)
        x2, y2 = x1 + w, y1 + h
        bg[y1:y2, x1:x2] = np.where(patch_mask[..., None] > 0, patch, bg[y1:y2, x1:x2])
        full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = np.where(patch_mask > 0, 1, 0)
        return bg, full_mask

    
    def _crop_mask_and_patch(self, mask: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop the mask and patch to the bounding box of the mask.
        """
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None, None
        y1, y2, x1, x2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
        patch_cropped = patch[y1:y2+1, x1:x2+1]
        mask_cropped = mask[y1:y2+1, x1:x2+1]
        return mask_cropped, patch_cropped

    def get_vis_img(self, img, mask, referring_text):
        vis_img = img.copy()
        if vis_img.max() <= 1.0:
            vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # Create colored mask for target instance (green)
        target_mask = (mask > 0).astype(np.uint8) * 255
        target_mask_colored = np.zeros_like(vis_img)
        target_mask_colored[:, :, 1] = target_mask  # Green channel

        # Overlay
        overlay = cv2.addWeighted(vis_img, 0.6, target_mask_colored, 0.4, 0)

        # Add text
        # cv2.putText(overlay, referring_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * vis_img.shape[1] / 640, (255, 255, 255), 2)
        font_scale = max(0.4, min(1.5, (vis_img.shape[1] / 640) * (25 / len(referring_text))))
        thickness = int(font_scale * 2)
        text_size = cv2.getTextSize(referring_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] 
        x = min(10, vis_img.shape[1] - text_size[0] - 10)
        cv2.putText(overlay, referring_text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, thickness), lineType=cv2.LINE_AA)
        return overlay
        
    
    def tokenize_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]: 
        """
        Tokenize the input text and return padded input IDs and attention mask.
        
        Args:
            text (str): Input text to tokenize.
        
        Returns:
            tuple: Padded input IDs and attention mask as tensors.
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        if len(encoded) > self.max_tokens:
            encoded = encoded[:self.max_tokens]
        padding_length = self.max_tokens - len(encoded)
        padded_ids = encoded + [0] * padding_length
        attention_mask = [1] * len(encoded) + [0] * padding_length
        
        return torch.tensor(padded_ids).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)
    
    def create_scrambled_background_from_single_image(self, bg_idx=None, rows=16, cols=16, blur_kernel_ratio=0.02):
        """
        从一张完整图像中裁剪 rows*cols 个 patch，打乱后重新拼接为同尺寸背景图。
        
        Args:
            bg_idx (int or None): 背景图像索引，None 表示随机选择
            rows (int): 行数
            cols (int): 列数
            blur_kernel_ratio (float): 模糊核大小占 min(H,W) 的比例

        Returns:
            bg (np.ndarray): (H, W, 3) uint8，打乱后的背景，尺寸与原图相同
            bg_brightness (float): 背景平均亮度，用于前景亮度匹配
        """
        # === Step 1: 加载背景图像 ===
        if bg_idx is None:
            bg_idx = np.random.randint(0, len(self.index))
        data = self.load(bg_idx)
        bg_img = data['img']  # (H, W, 3)
        h, w = bg_img.shape[:2]
        
        # kernel_size = int(blur_kernel_ratio * min(h, w))
        # kernel_size = kernel_size // 2 * 2 + 1  # 奇数
        # bg_img = cv2.blur(bg_img, (kernel_size, kernel_size))

        total_patches = rows * cols
        patch_h = h // rows
        patch_w = w // cols


        # === Step 2: 从原图中规则或随机裁剪 patch ===
        patches = []
        for _ in range(total_patches):
            # 随机裁剪略小于目标尺寸的 patch（增加多样性）
            dh = random.randint(int(0.9 * patch_h), patch_h)
            dw = random.randint(int(0.9 * patch_w), patch_w)
            y = np.random.randint(0, h - dh)
            x = np.random.randint(0, w - dw)
            patch = bg_img[y:y+dh, x:x+dw]
            patches.append(patch)

        # === Step 3: 打乱 patch 顺序 ===
        random.shuffle(patches)

        # === Step 4: 拼接成 rows × cols 网格，恢复为原图大小 ===
        avg_color = bg_img.mean(axis=(0, 1))
        reconstructed = np.ones((h, w, 3), dtype=np.uint8) * avg_color.astype(np.uint8)
        for i in range(rows):
            for j in range(cols):
                patch = patches.pop()
                ph, pw = patch.shape[:2]
                y1 = i * patch_h
                x1 = j * patch_w
                y2 = y1 + ph
                x2 = x1 + pw
                # 缩放 patch 到目标格子大小（允许形变）
                resized_patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
                reconstructed[y1:y1+patch_h, x1:x1+patch_w] = resized_patch
            if not patches:
                break  # 用完就停止


        # === Step 5: 模糊融合，使拼接更自然 ===
        kernel_size = int(blur_kernel_ratio * min(h, w))
        kernel_size = kernel_size // 2 * 2 + 1
        bg = cv2.blur(reconstructed, (kernel_size, kernel_size))
        # === Step 6: 计算亮度 ===
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        bg_brightness = bg_gray.mean()

        return bg, bg_brightness


if __name__ == "__main__":
    # Example usage
    dataset = SynthesisDataset(prob=0.5, root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco", dataset='unc', split='train')
    print(f"Dataset length: {len(dataset)}")
    
    # Load a sample
    for i in range(10):
        sample = dataset.load(0)
        print(f"Sample text: {sample['txt']}")
        
        # Tokenize text
        input_ids, attention_mask = dataset.tokenize_text(sample['txt'])
        print(f"Input IDs: {input_ids}, Attention Mask: {attention_mask}")
        
        # Create scrambled background
        bg, bg_brightness = dataset.create_scrambled_background_from_single_image(rows=16, cols=16, blur_kernel_ratio=0.04)
        print(f"Background shape: {bg.shape}, Brightness: {bg_brightness}")
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"visualizations/synthetics/scrambled_background_{i}.jpg", bg)
    
    
    



    
    
    
    


        