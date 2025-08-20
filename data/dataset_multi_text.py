import numpy as np
import os
import json
from data.common import AbstractDataset


class MultiTextDataset(AbstractDataset):
    
    def __init__(self, root: str = "/data/datasets/tzhangbu/Cherry-Pick/data/refcoco", 
                 cls_result_root: str = "/data/datasets/tzhangbu/Cherry-Pick/local_trial",
                 dataset: str = "unc", 
                 split = "train", 
                 max_tokens=20,
                 max_iters = None,
                 image_transforms=None):
        
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
        
        print("==" * 20)
        print(f"Loading dataset from {self.index_root}")
        print(f"Image text ground truth root: {self.image_txt_gt_root}")
        print(f"Mask root: {self.mask_root}")
        print(f"Classification results path: {self.cls_results_path}")
        print("Find {} images and {} targets".format(len(self.img_items), len(self.target_items)))
        print("single target count: ", single_count)
        print("==" * 20)
        
            
    def __len__(self):
        return len(self.target_items)

    def __getitem__(self, idx):
        
        target_item = self.target_items[idx]
        ## Currently, randomly select a referring text result
        selected_index = np.random.choice(target_item)
        ## Select another text from the same target groups, 
        others = [i for i in target_item if i != selected_index]
        aug_index = np.random.choice(others)
        
        index_path = os.path.join(self.index_root, f"{self.dataset}_{self.split}_{selected_index}.json")
        aug_index_path = os.path.join(self.index_root, f"{self.dataset}_{self.split}_{aug_index}.json")
        
        raw_img, raw_target_array, txt = self.load_from_index(index_path)
        
        ## Augment text
        aug_index_data = json.load(open(aug_index_path, 'r'))
        aug_img_txt_gt_name = aug_index_data["img_txt_gt_file_name"]
        aug_img_txt_gt_path = os.path.join(self.image_txt_gt_root, aug_img_txt_gt_name)
        aug_img_txt_gt = np.load(aug_img_txt_gt_path, allow_pickle=True)
        aug_data_dict = {key: aug_img_txt_gt[key] for key in aug_img_txt_gt}
        aug_txt = aug_data_dict['sent_batch'][0]
        # if aug_txt == txt:
        #     print(f"Warning: Augmented text is the same as original text for index {idx}. Using original text.")
        #     print(f"Aug index: {aug_index}")
        #     print(f"index: {selected_index}")

        img, target = self.apply_transform(raw_img.copy(), raw_target_array.copy())
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
    dataset = MultiTextDataset()
    sample_counts = 1000
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
    
    
