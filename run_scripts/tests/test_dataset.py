from misc.common import make_object_from_config
import json
from torch.utils.data import DataLoader

if __name__ == "__main__":
    config_root = "configs/dataset_config_example1.json"
    configs = json.load(open(config_root, 'r'))

    dataset = make_object_from_config(configs["dataset_configs"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    import tqdm
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        img, mask, input_ids, attention_mask = data
        print(f"Batch {i}:")
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_mask.shape}")
    

    