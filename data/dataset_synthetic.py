import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from bert.tokenization_bert import BertTokenizer
import os
import re
import json
from pycocotools import mask as pycocotools_mask
import torch
import abc
from abc import ABC, abstractmethod
import transforms as T
import cv2
from data.utils import has_spatial_expression

class SynthesisDataset:
    
    def __init__(self, 
                 prob: float, 
                 root: str, 
                 dataset: str,
                 split: str,
                 max_tokens: int = 20, 
                 load_raw_data: bool = False,
                 **kwargs):
        self.prob = prob
        self.max_tokens = max_tokens
        self.load_raw_data = load_raw_data
        
        self.root = root
        self.dataset = dataset
        self.split = split
        
        self.index_root = f"{self.root}/{self.dataset}/{self.split}_purified_mask_list.json"
        self.image_txt_gt_root = f"{self.root}/{self.dataset}/{self.split}_batch"
        self.pseudo_label_root = f"{self.root}/{self.dataset}/{self.split}_mask_newB_batch"
        self.noun_dict_path = f"{self.root}/{self.dataset}/{self.dataset}_noun/{self.dataset}_{self.split}_dict.npy"
        
        assert os.path.exists(self.index_root), f"Index file {self.index_root} does not exist."
        assert os.path.exists(self.image_txt_gt_root), f"Image and text ground truth root {self.image_txt_gt_root} does not exist."
        assert os.path.exists(self.pseudo_label_root), f"Pseudo label root {self.pseudo_label_root} does not exist."
        assert os.path.exists(self.noun_dict_path), f"Noun dictionary {self.noun_dict_path} does not exist."
        
        with open(self.index_root, 'r') as f:
            self.index = json.load(f)
        
        with open(self.noun_dict_path, 'rb') as f:
            self.noun_dict = np.load(f, allow_pickle=True).item()
        
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

class RowColumnOrdinalDataset(SynthesisDataset):
    
    def __init__(self, prob: float, root: str, dataset: str, split: str, max_tokens: int = 20, range_num: tuple = (1,4), **kwargs):
        super().__init__(prob, root, dataset, split, max_tokens, **kwargs)
        self.range_num = range_num  # (min, max) total number of instances
        self.bg_color = (128, 128, 128)  # gray background
        
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
    
    def load_until_success(self) -> dict:
        max_attempts = 10
        for _ in range(max_attempts):
            idx = np.random.randint(0, len(self.index))
            data = self.load(idx)
            if data is not None and not has_spatial_expression(data['noun']):
                return data
        return data
    
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
        
        
    
    def __call__(self, idx=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx is None:
            idx = np.random.randint(0, len(self.index))
        # data = self.load(idx)
        data = self.load_until_success()
        img_array = data['img']
        noun = data['noun'] 
        mask = data['mask']
        patch_mask, patch = self._crop_mask_and_patch(mask, img_array)
        expansion_factor = np.random.uniform(1.0, 1.5)
        cell_w, cell_h = patch.shape[1] * expansion_factor, patch.shape[0] * expansion_factor
        # Determine number of objects
        cols = np.random.randint(self.range_num[0], self.range_num[1] + 1)
        rows = np.random.randint(self.range_num[0], self.range_num[1] + 1)
        num = rows * cols
        
        target_idx = np.random.randint(0, num)  # 被指代的是第几个
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                min_x1 = int(j * cell_w)
                max_x1 = int((j + (1.0 - 1.0 / expansion_factor)) * cell_w)
                min_y1 = int(i * cell_h)
                max_y1 = int((i + (1.0 - 1.0 / expansion_factor)) * cell_h)
                max_x1 = max(min_x1 + 1, max_x1) 
                max_y1 = max(min_y1 + 1, max_y1)
                x1 = np.random.randint(min_x1, max_x1)
                y1 = np.random.randint(min_y1, max_y1)
                
                positions.append(
                    {
                        "index_position": (i, j),
                        "position": (x1, y1),
                    }
                )
        use_real_bg = True
        if use_real_bg:
            # === Step 1: 随机选择一个与当前样本不同的背景图像 ===
            bg_idx = np.random.randint(0, len(self.index))
            bg_data = self.load(bg_idx)
            bg = bg_data['img']  # (H, W, 3), uint8

            # === Step 2: 调整背景尺寸以匹配合成画布 ===
            target_w = int(cols * cell_w)
            target_h = int(rows * cell_h)
            bg = cv2.resize(bg, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            kernel_size = max(31, int(min(cell_w, cell_h) * 0.2) // 2 * 2 + 1)  # 必须为奇数
            bg = cv2.blur(bg, (kernel_size, kernel_size))
            
            # === Step 4: 可选 - 调整 patch 亮度以匹配背景亮度（视觉融合更好） ===
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
            bg_brightness = bg_gray.mean()

            # 计算 patch 亮度
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            patch_brightness = patch_gray.mean()

            # 调整 patch 亮度到接近背景
            brightness_ratio = bg_brightness / (patch_brightness + 1e-5)
            adjusted_patch = np.clip(patch.astype(np.float32) * brightness_ratio, 0, 255).astype(np.uint8)
            patch = adjusted_patch  # 替换为亮度匹配后的 patch

            # === Step 5: 继续使用 self.paste 将 patch 贴到模糊背景上 ===
        else:
            # 使用原灰色背景
            bg = np.full((int(rows * cell_h), int(cols * cell_w), 3), self.bg_color, dtype=np.uint8)
            full_mask = np.zeros((rows * int(cell_h), cols * int(cell_w)), dtype=np.uint8)

        for i, data in enumerate(positions):
            x, y = data["position"]
            row, col = data["index_position"]
            bg, full_obj_mask = self.paste(bg, patch, patch_mask, x, y)
            if i == target_idx:
                full_mask = full_obj_mask 
                row_idx = row + 1
                col_idx = col + 1
        
        # Ordinal list (up to 10)
        ordinals = ["first", "second", "third", "fourth", "fifth",
                    "sixth", "seventh", "eighth", "ninth", "tenth"]

        # ========================================================
        # 1. Basic Index-Based Templates (Clear and Direct)
        # ========================================================
        templates = []

        # 1. Basic Index-Based Templates
        templates += [
            f"{noun} in row {row_idx} column {col_idx}",
            f"the {noun} in row {row_idx}, column {col_idx}",
            f"find the {noun} at position ({row_idx}, {col_idx})",
            f"select the {noun} in row {row_idx} and column {col_idx}",
            f"the {noun} located at row {row_idx}, column {col_idx}",
            f"look for the {noun} in row {row_idx} and column {col_idx}",
            f"the {noun} that is in row number {row_idx} and column number {col_idx}",
        ]

        # 2. Ordinal Language Templates (Higher Priority!)
        if row_idx <= 10 and col_idx <= 10:
            templates += [
                f"the {noun} in the {ordinals[row_idx-1]} row and {ordinals[col_idx-1]} column",
                f"the {ordinals[row_idx-1]} row, {ordinals[col_idx-1]} column {noun}",
                f"the {noun} {ordinals[row_idx-1]} from the top and {ordinals[col_idx-1]} from the left",
                f"the {noun} that is {row_idx} rows down and {col_idx} columns across",
                f"the {row_idx} rows down and {col_idx} columns right to find the {noun}",
            ]

        # 3. Grid/Table Context
        templates += [
            f"{noun} is in row {row_idx}, column {col_idx}",
            f"the {noun} sits in the {ordinals[row_idx-1]} row and {ordinals[col_idx-1]} column",
            f"the {noun} occupies cell ({row_idx}, {col_idx}) in a {rows}×{cols} layout",
            f"{row_idx}, {col_idx} positions"
        ]
        if row_idx <= 10 and col_idx <= 10:
            templates.append(f"the {noun} in the {ordinals[row_idx-1]} row and {ordinals[col_idx-1]} column")

        # 4. Directional & Counting Instructions
        if row_idx <= 10 and col_idx <= 10:
            templates += [
                f"starting from the top-left, go down {row_idx-1} {'row' if row_idx == 1 else 'rows'} and right {col_idx-1} {'column' if col_idx == 1 else 'columns'} to find the {noun}",
                f"move {row_idx} rows down and {col_idx} columns from the left to reach the {noun}",
            ]

        # 5. Corner Templates
        if row_idx == 1 and col_idx == 1:
            templates += [f"top-left {noun}", f"the {noun} in the top-left corner"]
        elif row_idx == 1 and col_idx == cols:
            templates += [f"top-right {noun}", f"the {noun} in the top-right corner"]
        elif row_idx == rows and col_idx == 1:
            templates += [f"bottom-left {noun}", f"the {noun} in the bottom-left corner"]
        elif row_idx == rows and col_idx == cols:
            templates += [f"bottom-right {noun}", f"the {noun} in the bottom-right corner"]

        # 6. Edge (Boundary) Templates
        if row_idx == 1 and col_idx <= 10:
            templates.append(f"the {noun} in the topmost row and {ordinals[col_idx-1]} column")
        if row_idx == rows and col_idx <= 10:
            templates.append(f"the {noun} in the bottom row and {ordinals[col_idx-1]} column")
        if col_idx == 1 and row_idx <= 10:
            templates.append(f"the {noun} in the leftmost column and {ordinals[row_idx-1]} row")
        if col_idx == cols and row_idx <= 10:
            templates.append(f"the {noun} in the rightmost column and {ordinals[row_idx-1]} row")

        # 7. Center & Near-Center Templates
        center_row = (rows + 1) // 2
        center_col = (cols + 1) // 2
        is_odd_rows = (rows % 2 == 1)  # 只有奇数行才有唯一中间行
        is_odd_cols = (cols % 2 == 1)  # 只有奇数列才有唯一中间列

        templates = []  # 假设 templates 已定义，否则改为 +=

        # ========================================================
        # 1. "Middle" / "Center" — 仅在奇数行列时使用（保证唯一性）
        # ========================================================
        if is_odd_rows and is_odd_cols and row_idx == center_row and col_idx == center_col:
            templates += [
                f"the {noun} in the middle",
                f"the {noun} at the center",
                f"the central {noun}",
                f"the middle {noun}",
                f"center {noun}",
            ]
        # 如果不是奇数，但仍在几何中心附近，仍可用 "center" 但避免 "middle"
        elif row_idx == center_row and col_idx == center_col:
            templates += [
                f"the {noun} at the center",
                f"the central {noun}",
                f"center {noun}",
            ]

        # ========================================================
        # 2. "Near Center" → 细化为方向性描述（避免歧义）
        # ========================================================
        dr = row_idx - center_row  # 偏移行：-1=above, +1=below
        dc = col_idx - center_col  # 偏移列：-1=left, +1=right

        if 1 <= abs(dr) <= 1 and 1 <= abs(dc) <= 1 and not (dr == 0 and dc == 0):
            # 使用方向组合描述
            vert = "above" if dr < 0 else "below" if dr > 0 else ""
            horiz = "to the left of" if dc < 0 else "to the right of" if dc > 0 else ""

            if vert and horiz:
                templates.append(f"the {noun} {vert} and {horiz} the center")
                templates.append(f"the {noun} {horiz} and {vert} the middle")
            elif vert:
                templates.append(f"the {noun} just {vert} the center")
                templates.append(f"the {noun} slightly {vert} of the middle")
            elif horiz:
                templates.append(f"the {noun} just {horiz} the center")
                templates.append(f"the {noun} slightly {horiz} of the middle")

            # 通用 fallback
            # templates.append(f"the {noun} close to the center")
            # templates.append(f"the {noun} near the middle")

        # ========================================================
        # 3. Centered in One Axis Only
        # ========================================================
        if is_odd_rows and row_idx == center_row and col_idx != center_col and col_idx <= 10:
            templates.append(f"centered vertically, in the {ordinals[col_idx-1]} column {noun}")
            templates.append(f"on the vertical center line, in the {ordinals[col_idx-1]} column {noun}")

        if is_odd_cols and col_idx == center_col and row_idx != center_row and row_idx <= 10:
            templates.append(f"centered horizontally, in the {ordinals[row_idx-1]} row {noun}")
            templates.append(f"on the horizontal midline, in the {ordinals[row_idx-1]} row {noun}")
        # ========================================================
        # ✅ 新增：Middle 描述（适用于小网格，更自然）
        # ========================================================
        if rows == 3 and row_idx == 2:
            if cols == 3 and col_idx == 2:
                templates.append(f"the {noun} in the middle row and middle column")
            elif col_idx <= 10:
                templates.append(f"the {noun} in the middle row and {ordinals[col_idx-1]} column")
            else:
                templates.append(f"the {noun} in the middle row, column {col_idx}")

        if cols == 3 and col_idx == 2:
            if row_idx <= 10:
                templates.append(f"the {noun} in the {ordinals[row_idx-1]} row and middle column")
            else:
                templates.append(f"the {noun} in row {row_idx} and middle column")

        if rows == 2 and row_idx == 1:
            templates.append(f"the {noun} in the top row")
        if rows == 2 and row_idx == 2:
            templates.append(f"the {noun} in the bottom row")
        if cols == 2 and col_idx == 1:
            templates.append(f"the {noun} in the left column")
        if cols == 2 and col_idx == 2:
            templates.append(f"the {noun} in the right column")

        # ========================================================
        # 9. Question & Dialogue Style
        # ========================================================
        templates += [
            f"Which {noun} is in row {row_idx}, column {col_idx}? It's this one.",
            f"Can you find the {noun} in the {ordinals[row_idx-1]} row and {ordinals[col_idx-1]} column?",
        ]
        if row_idx <= 10 and col_idx <= 10:
            templates.append(f"Where is the {noun} in the {ordinals[row_idx-1]} row and {ordinals[col_idx-1]} column?")

        # ========================================================
        # ✅ 最终选择：加权采样，提升序数词模板权重
        # ========================================================
        weights = []

        for t in templates:
            weight = 1.0  # base weight
            # 如果包含序数词，提升权重
            if any(f"the {ord}" in t for ord in ordinals[:10]) or \
               any(f"{ord} " in t for ord in ordinals[:10]) or \
               "from the top" in t or "from the left" in t:
                weight *= 2.0  # 双倍权重

            # 特别鼓励 middle row/column（自然表达）
            if "middle row" in t or "middle column" in t:
                weight *= 1.5

            # 鼓励 corner 和 center 直接描述
            if any(kw in t for kw in ["top-left", "bottom-right", "center", "middle"]):
                weight *= 1.2

            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # 可选：用于调试
        text = np.random.choice(templates, p=weights / weights.sum())
        
        # Padding
        mean_value = np.mean(bg)
        bg = self.add_padding(bg, target_aspect=1.0, pad_value=mean_value)
        full_mask = self.add_padding(full_mask, target_aspect=1.0, pad_value=0)


        # Finalize
        if not self.load_raw_data:
            full_mask_img = Image.fromarray(full_mask.astype(np.uint8)).convert("P")
            img_pil = Image.fromarray(bg.astype(np.uint8)).convert("RGB")
            img_tensor, mask_tensor = self.apply_transforms(img_pil, full_mask_img)
            input_ids, attention_mask = self.tokenize_text(text)
            return img_tensor, mask_tensor, input_ids, attention_mask
        else:
            return bg, text, full_mask  # full_mask 是二值 mask，只包含目标 instance


if __name__ == "__main__":
    import os
    import cv2
    import numpy as np

    os.makedirs("visualizations/synthetics", exist_ok=True)

    dataset = RowColumnOrdinalDataset(
        prob=1.0,
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc", 
        split="train", 
        max_tokens=20, 
        n_objects=(3, 6), 
        load_raw_data=True
    )

    print("== Testing RowColumnOrdinalDataset (single-instance referring) ==")

    for i in range(100):
        result = dataset()
        if result is None:
            print(f"[{i+1}] Failed to generate sample.")
            continue

        img, txt, mask = result  # mask 是二值的，只包含目标 instance

        print(f"Referring text: {txt}")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")  # 应该是 [0, 1]

        # Prepare image
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
        cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * vis_img.shape[1] / 640, (255, 255, 255), 2)

        # Save
        save_path = f"visualizations/synthetics/referring_example_{i+1}.png"
        cv2.imwrite(save_path, overlay)
        print(f"Saved to {save_path}\n")
    
    
    



    
    
    
    


        