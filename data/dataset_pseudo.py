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
                 max_tokens=20,
                 augment_text_root="augmentation/data/unc/train"):
        self.root = root
        self.dataset = dataset
        self.split = split
        
        self.index_root = f"{self.root}/{self.dataset}/{self.split}_pseudo_score"
        self.image_txt_gt_root = f"{self.root}/{self.dataset}/{self.split}_batch"
        self.mask_root = f"{self.root}/{self.dataset}/{self.split}_mask_newB_batch"
        self.augment_text_root = augment_text_root
    
        # Read and sort JSON files by number at the end of filename
        json_files = [f for f in os.listdir(self.index_root) if f.endswith('.json')]

        json_files_sorted = sorted(json_files, key=self.extract_number)
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
        
        ## Augment text
        data_id = self.extract_number(os.path.basename(index_path))
        augment_text_path = os.path.join(self.augment_text_root, f"{self.dataset}_{self.split}_augtext_{data_id}.json")
        if os.path.exists(augment_text_path):
            aug_data = json.load(open(augment_text_path, 'r'))
            aug_text_keys = list(aug_data.keys())[1:]
            if aug_text_keys is None or len(aug_text_keys) == 0:
                aug_txt = txt  # Fallback to original text if no augmented texts are available
            else:
                ## Random select one of the augmented texts
                selected = np.random.choice(list(aug_text_keys))
                aug_txt = aug_data[selected]
        else:
            aug_txt = txt
        
        mask_path = os.path.join(self.mask_root, mask_file_name)
        mask_candidates = json.load(open(mask_path, 'r'))["annotation"]
        rle_mask = mask_candidates[predicted_mask_id]["rle"]
        mask = pycocotools_mask.decode(rle_mask)
        mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
        img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        img, target = self.image_transforms(img, mask)
        
        padded_input_ids, attention_mask = self.tokenize_text(txt)
        try:
            aug_padded_input_ids, aug_attention_mask = self.tokenize_text(aug_txt)
        except Exception as e:
            print(f"Error tokenizing augmented text: {e}")
            aug_padded_input_ids, aug_attention_mask = self.tokenize_text(txt)
        return img, target, padded_input_ids, attention_mask, aug_padded_input_ids, aug_attention_mask
    
    def extract_number(self, filename):
        match = re.search(r'_(\d+)\.json$', filename)
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
    
    def get_raw_item(self, idx):
        """
        Get the raw item of the text, img, and mask and augmented text from the dataset.
        """
        if idx < 0 or idx >= len(self.index_list):
            raise IndexError("Index out of range.")
        index_path = self.index_list[idx]
        index_data = json.load(open(index_path, 'r'))
        img_tx_gt_name = index_data["img_txt_gt_file_name"]
        mask_file_name = index_data["mask_file_name"]
        predicted_mask_id = index_data["predicted_mask_id"]
        img_txt_gt_path = os.path.join(self.image_txt_gt_root, img_tx_gt_name)
        img_txt_gt = np.load(img_txt_gt_path, allow_pickle=True)
        data_dict = {key: img_txt_gt[key] for key in img_txt_gt}
        img = data_dict['im_batch']
        img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        txt = data_dict['sent_batch'][0]
        
        mask_path = os.path.join(self.mask_root, mask_file_name)
        mask_candidates = json.load(open(mask_path, 'r'))["annotation"]
        rle_mask = mask_candidates[predicted_mask_id]["rle"]
        mask = pycocotools_mask.decode(rle_mask)
        mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
        
        ## Augment text
        data_id = self.extract_number(os.path.basename(index_path))
        augment_text_path = os.path.join(self.augment_text_root, f"{self.dataset}_{self.split}_augtext_{data_id}.json")
        if os.path.exists(augment_text_path):
            aug_data = json.load(open(augment_text_path, 'r'))
            aug_text_keys = list(aug_data.keys())[1:]
            if aug_text_keys is None or len(aug_text_keys) == 0:
                aug_txt = txt  # Fallback to original text if no augmented texts are available
            else:
                ## Random select one of the augmented texts
                selected = np.random.choice(list(aug_text_keys))
                aug_txt = aug_data[selected]
        else:
            aug_txt = txt   
        
        return {
            "img": img,
            "txt": txt,
            "aug_txt": aug_txt,
            "mask": mask,
        }


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
    length = len(dataset)
    print(f"Dataset length: {length}")
    
    random_choices = np.random.choice(length, size=5, replace=False)
    print(f"Randomly selected indices: {random_choices}")
    
    ## Print some sample of text and augmented text
    
    for i in random_choices:
        data_dict = dataset.get_raw_item(i)
        print(f"Sample {i}:")
        print(f"Text: {data_dict['txt']}")
        print(f"Augmented Text: {data_dict['aug_txt']}")
        print(f"Image size: {data_dict['img'].size}")
        print(f"Mask size: {data_dict['mask'].size}")
        print("-" * 20)
    
    

        



    
    
        

