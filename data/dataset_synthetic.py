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
    
    def __init__(self, prob: float, root: str, dataset: str, split: str, max_tokens: int = 20,
                 layout: str = "row", n_objects: tuple = (3, 6), **kwargs):
        super().__init__(prob, root, dataset, split, max_tokens, **kwargs)
        assert layout in ["row", "column", "grid"], "layout must be 'row', 'column', or 'grid'"
        self.layout = layout
        self.n_objects = n_objects  # (min, max) total number of instances
        self.bg_color = (128, 128, 128)  # gray background
        
    def soft_paste(self, bg: np.ndarray, patch: np.ndarray, patch_mask: np.ndarray, x: int, y: int):
        ph, pw = patch.shape[:2]
        h, w = bg.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + pw), min(h, y + ph)

        if x1 >= x2 or y1 >= y2:
            return bg, np.zeros((h, w), dtype=np.uint8)

        # Crop patch and mask
        patch_cropped = patch[y1-y:y2-y, x1-x:x2-x]
        mask_cropped = patch_mask[y1-y:y2-y, x1-x:x2-x]

        # Soft blending
        mask_blurred = cv2.GaussianBlur(mask_cropped.astype(np.float32), (5, 5), 1)
        mask_soft = mask_blurred[..., None]

        roi = bg[y1:y2, x1:x2]
        blended = (patch_cropped * mask_soft + roi * (1 - mask_soft)).astype(np.uint8)
        bg[y1:y2, x1:x2] = blended

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_cropped
        return bg, full_mask
    
    def _get_grid_shape(self, n: int) -> tuple[int, int]:
        """
        Given total number of objects, return (rows, cols) that form a compact grid.
        Prefer wider grids (more columns).
        """
        if n == 1:
            return (1, 1)
        if n == 2:
            return (1, 2)
        
        # Find factor pairs (r, c) such that r * c >= n, and r,c as close as possible
        rows, cols = 1, n
        for r in range(1, int(np.sqrt(n)) + 2):
            c = (n + r - 1) // r  # ceil(n / r)
            if r * c >= n:
                if c <= cols and r <= c:  # prefer wider grids
                    rows, cols = r, c
        return rows, cols
    
    def __call__(self, idx=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx is None:
            idx = np.random.randint(0, len(self.index))
        data = self.load(idx)
        img_array = data['img']
        noun = data['txt']  # 假设是 "red bus" 这类名词短语
        mask = data['mask']

        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None
        y1, y2, x1, x2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
        patch = img_array[y1:y2+1, x1:x2+1]
        patch_mask = mask[y1:y2+1, x1:x2+1]

        # Create background
        H, W = 480, 480
        bg = np.full((H, W, 3), self.bg_color, dtype=np.uint8)
        full_mask = np.zeros((H, W), dtype=np.uint8)  # 二值 mask，只保留目标 instance

        # Determine number of objects
        num = np.random.randint(*self.n_objects)

        # Layout setup
        margin = 10
        H, W = 480, 480

        if self.layout == "row":
            x_positions = np.linspace(margin, W - margin, num).astype(int)
            y_center = H // 2 + np.random.randint(-20, 20)
            positions = [(x, y_center) for x in x_positions]
            obj_w = min(80, (W - 2 * margin) // max(num, 1) - 5)
            scale = obj_w / patch.shape[1]

        elif self.layout == "column":
            y_positions = np.linspace(margin, H - margin, num).astype(int)
            x_center = W // 2 + np.random.randint(-20, 20)
            positions = [(x_center, y) for y in y_positions]
            obj_h = min(80, (H - 2 * margin) // max(num, 1) - 5)
            scale = obj_h / patch.shape[0]

        else:  # grid
            rows, cols = self._get_grid_shape(num)
            x_positions = np.linspace(margin, W - margin, cols).astype(int)
            y_positions = np.linspace(margin, H - margin, rows).astype(int)
            positions = [(x, y) for y in y_positions for x in x_positions]
            obj_w = min(80, (W - 2 * margin) // max(cols, 1) - 5)
            obj_h = min(80, (H - 2 * margin) // max(rows, 1) - 5)
            scale_w = obj_w / patch.shape[1]
            scale_h = obj_h / patch.shape[0]
            scale = min(scale_w, scale_h)

        # Resize patch
        new_h, new_w = int(patch.shape[0] * scale), int(patch.shape[1] * scale)
        patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # === 随机选择一个目标 instance ===
        target_idx = np.random.randint(0, num)  # 被指代的是第几个
        target_position = None

        # Paste all instances (for context), but only mask the target
        for i, (x, y) in enumerate(positions):
            bg, obj_mask = self.soft_paste(bg, patch_resized, mask_resized, x - new_w // 2, y - new_h // 2)
            if i == target_idx:
                full_mask[obj_mask > 0] = 1  # 只保留目标 instance
                target_position = (x, y)

        # === 生成指向 target_idx 的文本 ===
        templates = []

        if self.layout == "row":
            side = np.random.choice(["left", "right"])
            ordinals = ["first", "second", "third", "fourth", "fifth", "sixth"]
            ord_str = ordinals[target_idx] if side == "left" else ordinals[num - target_idx - 1]
            templates = [
                f"the {ord_str} {noun} from the {side} in the row",
                f"{ord_str} {noun} in the row"
            ]

        elif self.layout == "column":
            side = np.random.choice(["top", "bottom"])
            ordinals = ["first", "second", "third", "fourth", "fifth", "sixth"]
            ord_str = ordinals[target_idx] if side == "top" else ordinals[num - target_idx - 1]
            templates = [
                f"the {ord_str} {noun} from the {side} in the column",
                f"{ord_str} {noun} in the column"
            ]

        else:  # grid
            rows, cols = self._get_grid_shape(num)
            row_idx = target_idx // cols + 1
            col_idx = target_idx % cols + 1
            ordinals = ["first", "second", "third", "fourth", "fifth", "sixth"]
            ord_str = ordinals[target_idx]

            # 2D 定位模板
            templates = [
                f"the {ord_str} {noun} in the grid",
                f"{noun} in row {row_idx} column {col_idx}",
                f"the {noun} in the {ordinals[row_idx-1]} row and {ordinals[col_idx-1]} column",
            ]
            if row_idx == 1 and col_idx == 1:
                templates.append(f"top-left {noun}")
            elif row_idx == 1 and col_idx == cols:
                templates.append(f"top-right {noun}")
            elif row_idx == rows and col_idx == 1:
                templates.append(f"bottom-left {noun}")
            elif row_idx == rows and col_idx == cols:
                templates.append(f"bottom-right {noun}")
            if row_idx == (rows + 1) // 2 and col_idx == (cols + 1) // 2:
                templates.append(f"center {noun}")

        text = np.random.choice(templates)

        print(f"Referring text: {text}")
        print(f"Target instance: #{target_idx + 1} / {num}")
        print(f"patch range: {patch_resized.min()} ~ {patch_resized.max()}, dtype: {patch_resized.dtype}")
        print(f"bg after paste range: {bg.min()} ~ {bg.max()}")

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
        layout="grid", 
        n_objects=(3, 6), 
        load_raw_data=True
    )

    print("== Testing RowColumnOrdinalDataset (single-instance referring) ==")

    for i in range(5):
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
        cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save
        save_path = f"visualizations/synthetics/referring_example_{i+1}.png"
        cv2.imwrite(save_path, overlay)
        print(f"Saved to {save_path}\n")
    
    
    



    
    
    
    


        