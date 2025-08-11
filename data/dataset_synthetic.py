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
    
    def __call__(self, idx=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx is None:
            idx = np.random.randint(0, len(self.index))
        data = self.load(idx)
        img_array = data['img']
        noun = data['txt'] 
        mask = data['mask']
        patch_mask, patch = self._crop_mask_and_patch(mask, img_array)
        cell_w, cell_h = patch.shape[1] * 2.0, patch.shape[0] * 2.0
        # Determine number of objects
        cols = np.random.randint(self.range_num[0], self.range_num[1] + 1)
        rows = np.random.randint(self.range_num[0], self.range_num[1] + 1)
        num = rows * cols
        
        target_idx = np.random.randint(0, num)  # 被指代的是第几个
        
        ## Generate positions (x1, y1) randomly
        positions = []
        for i in range(rows):
            for j in range(cols):
                min_x1 = int(j * cell_w)
                max_x1 = int((j + 0.5) * cell_w)
                min_y1 = int(i * cell_h)
                max_y1 = int((i + 0.5) * cell_h)
                x1 = np.random.randint(min_x1, max_x1)
                y1 = np.random.randint(min_y1, max_y1)
                
                positions.append(
                    {
                        "index_position": (i, j),
                        "position": (x1, y1),
                    }
                )
        
        bg = np.full((rows * int(cell_h), cols * int(cell_w), 3), self.bg_color, dtype=np.uint8)
        full_mask = np.zeros((rows * int(cell_h), cols * int(cell_w)), dtype=np.uint8)

        for i, data in enumerate(positions):
            x, y = data["position"]
            row, col = data["index_position"]
            bg, full_obj_mask = self.paste(bg, patch, patch_mask, x, y)
            if i == target_idx:
                full_mask = full_obj_mask 
                row_idx = row + 1
                col_idx = col + 1
    
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
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
    
    
    



    
    
    
    


        