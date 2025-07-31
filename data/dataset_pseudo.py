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


class PseudoLabelDataset(data.Dataset):
    
    def __init__(self, image_transforms, root: str = "/data/datasets/tzhangbu/Cherry-Pick/data/refcoco", 
                 dataset: str = "unc", 
                 split = "train", 
                 max_tokens=20):
        self.root = root
        self.dataset = dataset
        self.split = split
        
        self.index_root = f"{self.root}/{self.dataset}/{self.split}_pseudo_score"
        self.image_txt_gt_root = f"{self.root}/{self.dataset}/{self.split}_batch"
        self.mask_root = f"{self.root}/{self.dataset}/{self.split}_mask_newB_batch"
    
        # Read and sort JSON files by number at the end of filename
        json_files = [f for f in os.listdir(self.index_root) if f.endswith('.json')]
        def extract_number(filename):
            match = re.search(r'_(\d+)\.json$', filename)
            return int(match.group(1)) if match else -1
        json_files_sorted = sorted(json_files, key=extract_number)
        self.index_list = [os.path.join(self.index_root, f) for f in json_files_sorted]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_tokens = max_tokens
        self.image_transforms = image_transforms
            
        print(self.index_list[:10], "first 10 pseudo label files")
        print(len(self.index_list), "pseudo label files found")
    
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index_path = self.index_list[idx]
        index_data = json.load(open(index_path, 'r'))
        img_tx_gt_name = index_data["img_txt_gt_file_name"]
        mask_file_name = index_data["mask_file_name"]
        predicted_mask_id = index_data["predicted_mask_id"]
        
        img_txt_gt_path = os.path.join(self.image_txt_gt_root, img_tx_gt_name)
        img_txt_gt = np.load(img_txt_gt_path, allow_pickle=True)
        data_dict = {key: img_txt_gt[key] for key in img_txt_gt}
        img = data_dict['im_batch']
        txt = data_dict['sent_batch'][0]
        mask_path = os.path.join(self.mask_root, mask_file_name)
        mask_candidates = json.load(open(mask_path, 'r'))["annotation"]
        rle_mask = mask_candidates[predicted_mask_id]["rle"]
        mask = pycocotools_mask.decode(rle_mask)
        mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
        img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        img, target = self.image_transforms(img, mask)
        
        padded_input_ids = [0] * self.max_tokens
        input_ids = self.tokenizer.encode(txt, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]  # Truncate to max_token
        padded_input_ids[:len(input_ids)] = input_ids
        padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        padded_input_ids = padded_input_ids.unsqueeze(0)
        attention_mask = [1] * len(input_ids) + [0] * (self.max_tokens - len(input_ids))
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        
        return img, target, padded_input_ids, attention_mask
        

def get_dataset(root: str, dataset: str, split: str, image_transforms=None, max_tokens=20):
    """
    Get the PseudoLabelDataset.
    
    Args:
        root (str): Root directory of the dataset.
        dataset (str): Dataset name (e.g., 'unc').
        split (str): Split name (e.g., 'train').
        image_transforms: Image transformation function.
        max_tokens (int): Maximum number of tokens for text input.
        
    Returns:
        PseudoLabelDataset: The dataset instance.
    """
    if image_transforms is None:
        import transforms as T
        transforms = [T.Resize(480, 480),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]
        image_transforms = T.Compose(transforms)

    return PseudoLabelDataset(image_transforms, root, dataset, split, max_tokens)

if __name__ == "__main__":
    dataset = get_dataset(
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc",
        split="train",
        max_tokens=20
    )
    print(f"Dataset length: {len(dataset)}")
    item = dataset[0]
    
    print(f"Image shape: {item[0].shape}, Target shape: {item[1].shape}, Input IDs: {item[2].shape}, Attention Mask: {item[3].shape}")

        



    
    
        

