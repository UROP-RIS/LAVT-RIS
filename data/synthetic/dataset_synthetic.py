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


    
    
    



    
    
    
    


        