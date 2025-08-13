from data.synthetic.dataset_synthetic import SynthesisDataset
from data.synthetic.utils.spacy import has_spatial_expression 
from PIL import Image
from collections import deque
import numpy as np
import cv2
import torch
import random

min_gap = 10
overlap_tol = 20
small_offset = 5
large_offset = 30
RELATIONS = {
    "left": {
        "dx": (-float('inf'), -min_gap),  # A.x + A.w < B.x - min_gap
        "dy": (-overlap_tol, overlap_tol), # 垂直方向大致对齐
        "templates": [
            "{A} on the left of {B}",
            "the {A} to the left of the {B}",
            "{A} and {B}, with the {A} on the left"
        ]
    },
    "right": {
        "dx": (min_gap, float('inf')),
        "dy": (-overlap_tol, overlap_tol),
        "templates": [
            "{A} on the right of {B}",
            "the {A} beside the {B}, on its right",
            "{A} and {B}, with the {A} on the right"
        ]
    },
    "above": {
        "dy": (-float('inf'), -min_gap),
        "dx": (-overlap_tol, overlap_tol),
        "templates": [
            "{A} above the {B}",
            "the {A} positioned over the {B}",
            "the {B} with the {A} directly overhead",
            "the {A} sitting on top of the {B}",
            "the {A} hovering above the {B}",
            "the {A} placed above the {B}",
            "the {A} located above the {B}",
            "the {A} on {B}",
            "the {A} at the top of the {B}",
            "the {A} positioned at the top of the {B}",
            "the {A} situated at the top of the {B}",
            "the {A} placed at the top of the {B}",
            "the {A} lying at the top of the {B}",
            "the {A} resting at the top of the {B}",
            "the {A} positioned above the {B}",
            "the {A} situated above the {B}",
        ]
    },
    "below": {
        "dy": (min_gap, float('inf')),
        "dx": (-overlap_tol, overlap_tol),
        "templates": [
            "{A} below the {B}",
            "the {A} under the {B}",
            "the {A} beneath the {B}",
            "the {A} positioned under the {B}",
            "the {A} sitting below the {B}",
            "the {A} located below the {B}",
            "the {A} on the bottom of the {B}",
            "the {A} at the bottom of the {B}",
            "the {A} underneath the {B}",
            "the {A} positioned below the {B}",
            "the {A} situated below the {B}",
            "the {A} placed below the {B}",
            "the {A} lying below the {B}",
            "the {A} resting below the {B}",
            "the {A} positioned at the bottom of the {B}",
            "the {A} situated at the bottom of the {B}",
            "the {A} placed at the bottom of the {B}",
            "the {A} lying at the bottom of the {B}",
            "the {A} resting at the bottom of the {B}"
        ]
    },
    "behind": {
        "dx": (-small_offset, small_offset),
        "dy": (small_offset, large_offset),  # B slightly in front
        "z_order": "B_over_A",  # B 贴在 A 上面（视觉遮挡）
        "templates": [
            "the {A} behind the {B}",
            "the {A} at the back of the {B}",
            "the {A} positioned behind the {B}",
            "the {A} located behind the {B}",
            "the {A} situated at the back fo the {B}",
            
        ]
    },
    "in_front_of": {
        "dx": (-small_offset, small_offset),
        "dy": (-large_offset, -small_offset),
        "z_order": "A_over_B",
        "templates": [
            "the {A} in front of the {B}",
            "the {A} blocking the view of the {B}",
            "the {A} positioned in front of the {B}",
            "the {A} located in front of the {B}",
            "the {A} situated in front of the {B}",
            "the {A} at the front of the {B}",
            "the {A} standing in front of the {B}",
        ]
    },
    "near": {
    "distance": (0, 80),     
    "templates": [
        "the {A} near the {B}",
        "the {A} close to the {B}",
        "the {A} nearby the {B}",
        "the {A} in the vicinity of the {B}",
        "the {A} in proximity to the {B}",
        "the {A} adjacent to the {B}"
    ]
}
}

class RelativeDataset(SynthesisDataset):
    
    def __init__(self, prob: float, root: str, dataset: str, split: str, max_tokens: int = 20, **kwargs):
        super().__init__(prob, root, dataset, split, max_tokens, **kwargs)
        self.bg_color = (128, 128, 128)  # gray background

    def load_until_success(self) -> dict:
        max_attempts = 10
        for _ in range(max_attempts):
            idx = np.random.randint(0, len(self.index))
            data = self.load(idx)
            if data is not None and not has_spatial_expression(data['txt']):
                return data
        return data
    
    def load_objects(self, num_objects: int) -> list:
        """
        Load a list of objects from the noun dictionary.
        """
        objects = []
        for idx in range(num_objects):
            raw = self.load_until_success()
            crop_mask, crop_img = self._crop_mask_and_patch(raw["mask"], raw["img"])
            objects.append(
                {
                    "idx": idx,
                    "sentence": raw["txt"],
                    "noun": raw["noun"],
                    "mask": crop_mask,
                    "img": crop_img,
                }
            )
        return objects
    
    def select_anchor(self, objects):
        return random.choice(objects)
    
    def print_tree_structure(self, tree, idx_to_noun=None, prefix="", is_last=True):
        """
        美观地打印树结构。

        Args:
            tree: 树节点 {'idx': int, 'relation': str, 'children': [...]}
            idx_to_noun: dict, 将 idx 映射为 noun（可选，用于显示名称）
            prefix: 当前前缀（递归使用）
            is_last: 是否是父节点的最后一个孩子
        """
        if tree is None:
            return
        node_id = tree['idx']
        relation = tree['relation']
        children = tree['children']
        noun_str = f"({idx_to_noun.get(node_id, 'obj')})" if idx_to_noun else f"[idx:{node_id}]"
        connector = "└── " if is_last else "├── "
        rel_str = f" --{relation}--> " if relation else ""
        print(f"{prefix}{connector}{noun_str}{rel_str}")
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            extension = "    " if is_last else "│   "
            child_prefix = prefix + extension
            self.print_tree_structure(child, idx_to_noun, child_prefix, is_last_child)
    
    def build_tree_bfs(self, objects, min_children=1, max_children=3):
        """。
        
        Args:
            objects: List of object dicts [{'noun': 'cat', 'idx': 123}, ...]
            min_children: 每个节点最少孩子数
            max_children: 每个节点最多孩子数
        
        Returns:
            tree: dict with 'idx', 'relation', 'children'
        """
        if not objects:
            return None
        root_obj = random.choice(objects)
        unused_objects = set(obj['idx'] for obj in objects)  # 全局未使用 idx 集合
        unused_objects.discard(root_obj['idx'])
        tree = {
            'idx': root_obj['idx'],
            'relation': None,
            'children': []
        }
        if not unused_objects:
            return tree
        queue = deque()
        queue.append(tree)
        while queue:
            parent_node = queue.popleft()
            available_count = len(unused_objects)
            if available_count == 0:
                continue 
            num_children = min(available_count, 
                              random.randint(min_children, max_children))
            chosen_indices = random.sample(sorted(unused_objects), num_children)
            for idx in chosen_indices:
                relation = random.choice(list(RELATIONS.keys()))
                child_node = {
                    'idx': idx,
                    'relation': relation,
                    'children': []
                }
                parent_node['children'].append(child_node)
                unused_objects.remove(idx)
                queue.append(child_node)

        return tree
    
    def get_path_to_node(self, tree, target_idx):
        """
        返回从根到 target_idx 的节点路径（包含 relation）
        返回: [ {'idx': 100, 'relation': None}, {'idx': 201, 'relation': 'on'}, ... ]
        """
        path = []

        def dfs(node, current_path):
            current_path.append({
                'idx': node['idx'],
                'relation': node['relation']
            })
            if node['idx'] == target_idx:
                path.extend(current_path)
                return True
            for child in node['children']:
                if dfs(child, current_path):
                    return True
            current_path.pop()
            return False

        dfs(tree, [])
        return path
    
    def generate_referring_text(self, path, idx_to_noun):
        """
        递归生成 referring expression: 从根开始，逐层展开
        """
        if len(path) == 1:
            return f"the {idx_to_noun.get(path[0]['idx'], 'object')}"

        # 从根开始构建
        noun = idx_to_noun.get(path[0]['idx'], 'object')
        for i in range(1, len(path)):
            prev_noun = noun
            curr_idx = path[i]['idx']
            curr_noun = idx_to_noun.get(curr_idx, 'object')
            relation = path[i]['relation']
            
            template = random.choice(RELATIONS[relation]["templates"])
            # template: "{A} {rel} the {B}"
            noun = template.format(A=curr_noun, B=prev_noun)
        
        return noun

if __name__ == "__main__":
    dataset = RelativeDataset(
        prob=0.5, 
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc", 
        split="train", 
        max_tokens=20, 
        load_raw_data=True
    )
    
    # Example usage: 10 samples
    for i in range(10): 
        data = dataset.load_objects(6)
        trees = dataset.build_tree_bfs(data)
        idx_to_noun = {obj['idx']: obj['noun'] for obj in data}
        dataset.print_tree_structure(trees, idx_to_noun)
        referring_text = dataset.generate_referring_text(
            dataset.get_path_to_node(trees, data[1]['idx']), idx_to_noun
        )
        print(f"Referring expression: {referring_text}")
        
