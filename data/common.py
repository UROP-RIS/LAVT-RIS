import torch
import re
import transforms as T
from bert.tokenization_bert import BertTokenizer
import json
import os
import numpy as np
from pycocotools import mask as pycocotools_mask
from PIL import Image


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, dataset: str, split: str, max_tokens=20, index_suffix: str | None = None, image_transforms=None):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.image_transforms = image_transforms
        self.max_tokens = max_tokens
        if image_transforms is None:
            transforms = [T.Resize(480, 480),
                          T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ]
            self.image_transforms = T.Compose(transforms)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if index_suffix is None:
            self.index_root = f"{self.root}/{self.dataset}/{self.split}_pseudo_score"
            self.mt_label_root = f"{self.root}/{self.dataset}/{self.split}_mt_pseudo_label"
        else:
            self.index_root = f"{self.root}/{self.dataset}/{self.split}_pseudo_score_{index_suffix}"
            self.mt_label_root = f"{self.root}/{self.dataset}/{self.split}_mt_pseudo_label_{index_suffix}"
        self.image_txt_gt_root = f"{self.root}/{self.dataset}/{self.split}_batch"
        self.mask_root = f"{self.root}/{self.dataset}/{self.split}_mask_newB_batch"
        
        print("==" * 20)
        print(f"Loading dataset from {self.index_root}")
        print(f"Image text ground truth root: {self.image_txt_gt_root}")
        print(f"Mask root: {self.mask_root}")
        
    def extract_number(self, filename):
        match = re.search(r'_(\d+)\.[^.]+$', filename)
        return int(match.group(1)) if match else -1
    
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
    
    def load_from_index(self, index_path: str):
        index_data = json.load(open(index_path, 'r'))
        img_tx_gt_name = index_data["img_txt_gt_file_name"]
        mask_file_name = index_data["mask_file_name"]
        predicted_mask_id = index_data["predicted_mask_id"]
        similarity_score = index_data["similarity_score"]
        
        img_txt_gt_path = os.path.join(self.image_txt_gt_root, img_tx_gt_name)
        img_txt_gt = np.load(img_txt_gt_path, allow_pickle=True)
        data_dict = {key: img_txt_gt[key] for key in img_txt_gt}
        img = data_dict['im_batch']
        txt = data_dict['sent_batch'][0]
        
        mask_path = os.path.join(self.mask_root, mask_file_name)
        mask_candidates = json.load(open(mask_path, 'r'))["annotation"]
        rle_mask = mask_candidates[predicted_mask_id]["rle"]
        mask_array = pycocotools_mask.decode(rle_mask)
        return img, mask_array, txt, similarity_score, predicted_mask_id
    
    def apply_transform(self, img: np.ndarray, target: np.ndarray):
        target = Image.fromarray(target.astype(np.uint8)).convert("P") 
        img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        img, target = self.image_transforms(img, target)
        return img, target
        
    def normalize_to_softmax(self, data: list):
        tensor_data = torch.tensor([x if x is not None else float('nan') for x in data])
        mask = ~torch.isnan(tensor_data)
        valid = tensor_data[mask]
        valid_softmax = torch.softmax(valid, dim=0)
        result = torch.zeros_like(tensor_data)
        result.masked_scatter_(mask, valid_softmax)
        return result
    