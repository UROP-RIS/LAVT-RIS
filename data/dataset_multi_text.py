import numpy as np
import os
import json
from data.common import AbstractDataset
import torch


class MultiTextDataset(AbstractDataset):
    
    def __init__(self, root: str = "/data/datasets/tzhangbu/Cherry-Pick/data/refcoco", 
                 cls_result_root: str = "/data/datasets/tzhangbu/Cherry-Pick/local_trial",
                 dataset: str = "unc", 
                 split = "train", 
                 max_tokens=20,
                 max_iters = None,
                 image_transforms=None,
                 mode="normal"):
        
        super().__init__(
            root=root, 
            dataset=dataset, 
            split=split, 
            max_tokens=max_tokens, 
            image_transforms=image_transforms
        )
        
        self.cls_results_path = f"{cls_result_root}/img_cls_result_{self.dataset}_{self.split}.json"
        self.cls_results = json.load(open(self.cls_results_path, 'r'))["grouped_data"]
        ## Img list
        self.img_items = []
        ## target list
        self.target_items = []
        single_count = 0
        for img_item in self.cls_results:
            for target_item in img_item:
                self.target_items.append(target_item)
                if len(target_item) == 1:
                    single_count += 1
            self.img_items.append(img_item)
        
        if max_iters is not None:
            self.target_items = self.target_items[:max_iters]
        self.mode = mode
        
        print("==" * 20)
        print(f"Loading dataset from {self.index_root}")
        print(f"Image text ground truth root: {self.image_txt_gt_root}")
        print(f"Mask root: {self.mask_root}")
        print(f"Classification results path: {self.cls_results_path}")
        print("Find {} images and {} targets".format(len(self.img_items), len(self.target_items)))
        print("single target count: ", single_count)
        print(f"Dataset mode: {mode}")
        print("==" * 20)

        
            
    def __len__(self):
        return len(self.target_items)
    
    def _normalize_to_softmax(self, data: list):
        tensor_data = torch.tensor([x if x is not None else float('nan') for x in data])
        mask = ~torch.isnan(tensor_data)
        valid = tensor_data[mask]
        valid_softmax = torch.softmax(valid, dim=0)
        result = torch.zeros_like(tensor_data)
        result.masked_scatter_(mask, valid_softmax)
        return result

    def __getitem__(self, idx):
        
        target_item = self.target_items[idx]
        ## Currently, randomly select a referring text result
        if self.mode == "normal" or self.mode == "weighted":
            selected_index = np.random.choice(target_item)
        elif self.mode == "best":
            best_scores = []
            for item in target_item:
                index_path = os.path.join(self.index_root, f"{self.dataset}_{self.split}_{item}.json")
                raw_img, raw_target_array, txt, similarity_score, predicted_mask_id = self.load_from_index(index_path)
                softmax_score = self._normalize_to_softmax(similarity_score)
                best_score = softmax_score[predicted_mask_id].item()
                best_scores.append(best_score)
            
            # print(best_scores)
            selected_index = target_item[np.argmax(best_scores)]
            # print(f"selected index: {np.argmax(best_scores)}({selected_index})", )
        
        ## Select another text from the same target groups, 
        others = [i for i in target_item if i != selected_index]
        if len(others) == 0:
            # If no other texts are available, use the same text for augmentation
            aug_index = selected_index
        else:
            aug_index = np.random.choice(others)
        index_path = os.path.join(self.index_root, f"{self.dataset}_{self.split}_{selected_index}.json")
        aug_index_path = os.path.join(self.index_root, f"{self.dataset}_{self.split}_{aug_index}.json")
        raw_img, raw_target_array, txt, similarity_score, predicted_mask_id = self.load_from_index(index_path)
        ## Augment text
        aug_index_data = json.load(open(aug_index_path, 'r'))
        aug_img_txt_gt_name = aug_index_data["img_txt_gt_file_name"]
        aug_img_txt_gt_path = os.path.join(self.image_txt_gt_root, aug_img_txt_gt_name)
        aug_img_txt_gt = np.load(aug_img_txt_gt_path, allow_pickle=True)
        aug_data_dict = {key: aug_img_txt_gt[key] for key in aug_img_txt_gt}
        aug_txt = aug_data_dict['sent_batch'][0]

        img, target = self.apply_transform(raw_img.copy(), raw_target_array.copy())
        
        if self.mode == "weighted":
            normalized_score = self._normalize_to_softmax(similarity_score)
            predicted_target_score, _ = torch.max(normalized_score, dim=0)
            target = target * predicted_target_score
        padded_input_ids, attention_mask = self.tokenize_text(txt)
        try:
            aug_padded_input_ids, aug_attention_mask = self.tokenize_text(aug_txt)
        except Exception as e:
            print(f"Error tokenizing augmented text: {e}")
            aug_padded_input_ids, aug_attention_mask = self.tokenize_text(txt)
        # return img, target, padded_input_ids, attention_mask, aug_padded_input_ids, aug_attention_mask
        batch = {
            "img": img,
            # "raw_img": raw_img, 
            "target": target,
            # "raw_target": raw_target_array,
            "txt": padded_input_ids,
            "raw_txt": txt,
            "attention_mask": attention_mask,
            "aug_txt": aug_padded_input_ids,
            "raw_aug_txt": aug_txt,
            "aug_attention_mask": aug_attention_mask,
        }
        return batch

if __name__ == "__main__":
    dataset = MultiTextDataset(mode="best")
    sample_counts = 10
    duplicate_text_counts = 0
    choices = np.random.choice(len(dataset), 1000, replace=False)
    for idx in choices:
        batch = dataset[idx]
        text = batch["raw_txt"]
        aug_text = batch["raw_aug_txt"]
        
        if text == aug_text:
            # print(f"Warning: Original text and augmented text are the same for index {idx}.")
            duplicate_text_counts += 1
    
    print(f"Total samples checked: {sample_counts}, Duplicate texts found: {duplicate_text_counts}")
    
    
