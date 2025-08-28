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
                 augment_text_root="augmentation/data",
                 dataset: str = "unc", 
                 split = "train", 
                 max_tokens=20,
                 max_iters = None,
                 eval_mode=False):
        self.root = root
        self.dataset = dataset
        self.split = split
        
        self.index_root = f"{self.root}/{self.dataset}/{self.split}_pseudo_score"
        self.image_txt_gt_root = f"{self.root}/{self.dataset}/{self.split}_batch"
        self.mask_root = f"{self.root}/{self.dataset}/{self.split}_mask_newB_batch"
        self.augment_text_root = f"{augment_text_root}/{self.dataset}/{self.split}"
        
        print("==" * 20)
        print(f"Loading dataset from {self.index_root}")
        print(f"Image text ground truth root: {self.image_txt_gt_root}")
        print(f"Mask root: {self.mask_root}")
        print(f"Augment text root: {self.augment_text_root}")
        print("==" * 20)
    
        # Read and sort JSON files by number at the end of filename
        json_files = [f for f in os.listdir(self.index_root) if f.endswith('.json')]
        if max_iters is not None:
            json_files = json_files[:max_iters]

        json_files_sorted = sorted(json_files, key=self.extract_number)
        self.index_list = [os.path.join(self.index_root, f) for f in json_files_sorted]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_tokens = max_tokens
        self.image_transforms = image_transforms
        self.eval_mode = eval_mode
            
    
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index_path = self.index_list[idx]
        index_data = json.load(open(index_path, 'r'))
        img_tx_gt_name = index_data["img_txt_gt_file_name"]
        mask_file_name = index_data["mask_file_name"]
        predicted_mask_id = index_data["predicted_mask_id"]

        # Load image-text-ground truth
        img_txt_gt_path = os.path.join(self.image_txt_gt_root, img_tx_gt_name)
        img_txt_gt = np.load(img_txt_gt_path, allow_pickle=True)
        data_dict = {key: img_txt_gt[key] for key in img_txt_gt}
        raw_img = data_dict['im_batch']  # H x W x 3
        txt = data_dict['sent_batch'][0]

        orig_h, orig_w = raw_img.shape[:2]

        # Load mask candidates
        mask_path = os.path.join(self.mask_root, mask_file_name)
        mask_candidates = json.load(open(mask_path, 'r'))["annotation"]
        rle_mask = mask_candidates[predicted_mask_id]["rle"]
        raw_mask = pycocotools_mask.decode(rle_mask)  # H x W, binary

        # Convert to PIL for transforms
        img = Image.fromarray(raw_img.astype(np.uint8)).convert("RGB")
        mask = Image.fromarray(raw_mask.astype(np.uint8)).convert("P")

        # Apply transforms
        transformed_img, transformed_mask = self.image_transforms(img, mask)

        # Tokenize original and augmented text
        padded_input_ids, attention_mask = self.tokenize_text(txt)

        # Augmented text
        data_id = self.extract_number(os.path.basename(index_path))
        augment_text_path = os.path.join(self.augment_text_root, f"{self.dataset}_{self.split}_augtext_{data_id}.json")
        if os.path.exists(augment_text_path):
            aug_data = json.load(open(augment_text_path, 'r'))
            aug_text_keys = list(aug_data.keys())[1:]
            if aug_text_keys and len(aug_text_keys) > 0:
                selected = np.random.choice(aug_text_keys)
                aug_txt = aug_data[selected]
            else:
                aug_txt = txt
        else:
            aug_txt = txt

        # if aug_txt == txt:
        #     print(f"Warning: Augmented text is the same as original text for index {idx}. Using original text.")

        try:
            aug_padded_input_ids, aug_attention_mask = self.tokenize_text(aug_txt)
        except Exception as e:
            print(f"Error tokenizing augmented text: {e}")
            aug_padded_input_ids, aug_attention_mask = self.tokenize_text(txt)

        # Build base batch
        batch = {
            "img": transformed_img,
            "target": transformed_mask,
            "txt": padded_input_ids,
            "attention_mask": attention_mask,
            "aug_txt": aug_padded_input_ids,
            "aug_attention_mask": aug_attention_mask,
        }

        # Only in eval_mode: include raw data and extra info
        if self.eval_mode:
            # Convert raw mask to binary array for all candidates
            all_masks = []
            for candidate in mask_candidates:
                m = pycocotools_mask.decode(candidate["rle"])
                all_masks.append(m)  # each is H x W binary

            # GT mask
            gt_mask = data_dict["mask_batch"]  # assuming this is the ground truth

            batch.update({
                "raw_img": raw_img,           # original image array (H, W, 3)
                "raw_mask": raw_mask,         # predicted mask before transform (H, W)
                "all_masks": all_masks,       # list of all candidate masks (H, W)
                "gt": gt_mask,                # ground truth mask
                "orig_size": (orig_h, orig_w),# original size (H, W)
                "txt_raw": txt,               # original text string
                "aug_txt_raw": aug_txt,       # augmented text string
            })

        return batch

    @staticmethod
    def eval_collate_fn(batch):
        """
        Custom collate function for evaluation mode.
        - Stacks tensors that can be batched (img, target, txt, etc.)
        - Keeps raw data (raw_img, raw_mask, gt, etc.) as lists to avoid shape mismatch.
        """
        elem = batch[0]
        collated = {}
        keys = elem.keys()
        for key in keys:
            items = [d[key] for d in batch]
            if key in ['raw_img', 'raw_mask', 'gt', 'all_masks', 'txt_raw', 'aug_txt_raw', 'orig_size']:
                collated[key] = items
            else:
                collated[key] = torch.stack(items, dim=0)
        return collated
    
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
    


def get_dataset(root: str, augment_text_root: str, dataset: str, split: str, image_transforms=None, max_tokens=20, eval_mode=False, max_iters=None):
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

    return PseudoLabelDataset(image_transforms, root, augment_text_root, dataset, split, max_tokens, eval_mode=eval_mode, max_iters=max_iters)

if __name__ == "__main__":
    dataset = get_dataset(
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        augment_text_root="augmentation/data",
        dataset="unc+",
        split="train",
        max_tokens=20, 
        eval_mode=True
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
    
    

        



    
    
        

