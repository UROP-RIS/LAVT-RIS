from data.synthetic.dataset_synthetic import SynthesisDataset
from data.synthetic.utils.spacy import has_spatial_expression 
from data.synthetic.utils.misc import compute_iou_xywh, compute_iou_xyxy
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import numpy as np
import math
import cv2
import torch
import random

def visualize_layout(placed_obj, objects, tree, idx_to_noun, canvas_width, canvas_height, output_path="layout_visualization.png", bg_color=(230, 230, 230)):
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = bg_color
    # 使用高对比度颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    idx_to_color = {obj['idx']: colors[i % len(colors)] for i, obj in enumerate(objects)}

    centers = {}

    for obj in objects:
        idx = obj['idx']
        if idx not in placed_obj:
            continue

        pos = placed_obj[idx]
        x1, y1 = pos["x1y1"]
        x2, y2 = pos["x2y2"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x1 = np.clip(x1, 0, canvas_width - 1)
        y1 = np.clip(y1, 0, canvas_height - 1)
        x2 = np.clip(x2, 0, canvas_width - 1)
        y2 = np.clip(y2, 0, canvas_height - 1)

        if x1 >= x2 or y1 >= y2:
            continue

        color = idx_to_color[idx]
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)
        centers[idx] = ((x1 + x2) // 2, (y1 + y2) // 2)
        noun = idx_to_noun.get(idx, 'obj')
        cv2.putText(canvas, f"{idx}:{noun}", (x1, max(0, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def dfs_draw_edges(node):
        for child in node['children']:
            dfs_draw_edges(child)

    dfs_draw_edges(tree)

    cv2.imwrite(output_path, canvas)
    print(f"Saved to {output_path}")
    return canvas

min_gap = 10
overlap_tol = 20
small_offset = 5
large_offset = 30
RELATIONS = {
    "left": {
        "dx": {"mode": "factor", "range": (-3.0, -1.0)},
        "dy": {"mode": "abs", "range": (-overlap_tol, overlap_tol)},
        "templates": [
            "{A} on the left of {B}",
            "the {A} to the left of the {B}",
            "{A} and {B}, with the {A} on the left"
        ]
    },
    "right": {
        "dx": {"mode": "factor", "range": (1.0, 3.0)},
        "dy": {"mode": "abs", "range": (-overlap_tol, overlap_tol)},
        "templates": [
            "{A} on the right of {B}",
            "the {A} beside the {B}, on its right",
            "{A} and {B}, with the {A} on the right"
        ]
    },
    "above": {
        "dy": {"mode": "factor", "range": (-3.0, -1.0)},
        "dx": {"mode": "abs", "range": (-overlap_tol, overlap_tol)},
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
        "dy": {"mode": "factor", "range": (1.0, 3.0)},
        "dx": {"mode": "abs", "range": (-overlap_tol, overlap_tol)},
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
        "dx": {"mode": "abs", "range": (-small_offset, small_offset)},
        "dy": {"mode": "abs", "range": (small_offset, large_offset)},  # B slightly in front
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
        "dx": {"mode": "abs", "range": (-small_offset, small_offset)},
        "dy": {"mode": "abs", "range": (-large_offset, -small_offset)},
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
    "distance": {"mode": "factor", "range": (1.0, 2.0)},  
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
                    "w": crop_mask.shape[1],
                    "h": crop_mask.shape[0],
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


    def optimize_layout(self, tree, objects):
        placed_obj = {}
        idx_to_obj = {obj['idx']: obj for obj in objects}

        min_gap = 10
        max_retry_per_node = 5
        max_retry_root = 5
        iou_threshold = 0.1

        def get_bbox(cx, cy, w, h):
            return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

        def compute_iou(box1, box2):
            return compute_iou_xywh(
                [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2, box1[2] - box1[0], box1[3] - box1[1]],
                [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2, box2[2] - box2[0], box2[3] - box2[1]]
            )

        def sample_by_relation_strict(parent_obj, child_obj, relation):
            p_idx = parent_obj['idx']
            if p_idx not in placed_obj:
                return None, None, float('inf')
            px, py = placed_obj[p_idx]["cxcy"]
            pw, ph = parent_obj['w'], parent_obj['h']
            cw, ch = child_obj['w'], child_obj['h']
            parent_cx = px + pw / 2
            parent_cy = py + ph / 2

            best_max_iou = float('inf')
            best_pos = (parent_cx + 100, parent_cy + 100)
            attempts = 500

            for _ in range(attempts):
                cx, cy = None, None
                if relation == "left":
                    right_edge_max = parent_cx - pw / 2 - min_gap
                    left_edge_min = right_edge_max - 2 * max(pw, cw)
                    cx = random.uniform(left_edge_min + cw / 2, right_edge_max - cw / 2)
                    cy = random.uniform(parent_cy - ph * 0.3, parent_cy + ph * 0.3)
                elif relation == "right":
                    left_edge_min = parent_cx + pw / 2 + min_gap
                    right_edge_max = left_edge_min + 2 * max(pw, cw)
                    cx = random.uniform(left_edge_min + cw / 2, right_edge_max - cw / 2)
                    cy = random.uniform(parent_cy - ph * 0.3, parent_cy + ph * 0.3)
                elif relation == "above":
                    bottom_edge_max = parent_cy - ph / 2 - min_gap
                    top_edge_min = bottom_edge_max - 2 * max(ph, ch)
                    cy = random.uniform(top_edge_min + ch / 2, bottom_edge_max - ch / 2)
                    cx = random.uniform(parent_cx - pw * 0.3, parent_cx + pw * 0.3)
                elif relation == "below":
                    top_edge_min = parent_cy + ph / 2 + min_gap
                    bottom_edge_max = top_edge_min + 2 * max(ph, ch)
                    cy = random.uniform(top_edge_min + ch / 2, bottom_edge_max - ch / 2)
                    cx = random.uniform(parent_cx - pw * 0.3, parent_cx + pw * 0.3)
                elif relation == "near":
                    radius = 0.8 * math.sqrt(pw**2 + ph**2)
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(min_gap, radius)
                    cx = parent_cx + dist * math.cos(angle)
                    cy = parent_cy + dist * math.sin(angle)
                elif relation == "in_front_of":
                    offset = random.uniform(20, 100)
                    cx = parent_cx + random.uniform(-15, 15)
                    cy = parent_cy - offset
                elif relation == "behind":
                    offset = random.uniform(20, 100)
                    cx = parent_cx + random.uniform(-15, 15)
                    cy = parent_cy + offset
                else:
                    span = 2 * max(pw, ph)
                    cx = random.uniform(parent_cx - span, parent_cx + span)
                    cy = random.uniform(parent_cy - span, parent_cy + span)

                cx, cy = int(cx), int(cy)
                new_box = get_bbox(cx, cy, cw, ch)

                max_iou = 0.0
                valid = True
                for idx, pos in placed_obj.items():
                    if idx == child_obj['idx'] or idx == p_idx:
                        continue
                    ox, oy = pos["cxcy"]
                    other_w, other_h = objects[idx]['w'], objects[idx]['h']
                    other_box = [ox - other_w/2, oy - other_h/2, ox + other_w/2, oy + other_h/2]
                    iou = compute_iou(new_box, other_box)
                    if iou > iou_threshold:
                        valid = False
                    if iou > max_iou:
                        max_iou = iou

                if max_iou < best_max_iou:
                    best_max_iou = max_iou
                    best_pos = (cx, cy)

            best_x, best_y = best_pos
            return best_x, best_y, best_max_iou

        def dfs_traverse_children(node, current_obj):
            for child_node in node['children']:
                if not dfs_traverse(child_node, current_obj):
                    return False
            return True

        def dfs_traverse(node, parent_obj=None):
            obj_idx = node['idx']
            obj = idx_to_obj[obj_idx]

            if parent_obj is None:
                for _ in range(max_retry_root):
                    cx = random.randint(-200, 200)
                    cy = random.randint(-200, 200)
                    placed_obj[obj_idx] = {"cxcy": (cx, cy)}
                    if dfs_traverse_children(node, obj):
                        return True
                    del placed_obj[obj_idx]
                print(f"[Root] Failed to place root {obj['noun']} after {max_retry_root} retries.")
                return False

            relation = node['relation']
            last_best = None

            for retry in range(max_retry_per_node):
                x, y, max_iou = sample_by_relation_strict(parent_obj, obj, relation)
                last_best = (x, y, max_iou)

                if x is not None and max_iou <= iou_threshold:
                    placed_obj[obj_idx] = {"cxcy": (x, y)}
                    if dfs_traverse_children(node, obj):
                        return True
                    del placed_obj[obj_idx]
                    print(f"[Retry] Placed but children failed: {obj['noun']} at ({x}, {y}), attempt {retry + 1}")
                else:
                    print(f"[Retry] Failed to place {obj['noun']} with '{relation}' (attempt {retry + 1})")

            if last_best:
                x, y, max_iou = last_best
                placed_obj[obj_idx] = {"cxcy": (x, y)}
                print(f"[Fallback] Using best candidate for {obj['noun']} at ({x}, {y}) with IoU={max_iou:.3f}")
                if dfs_traverse_children(node, obj):
                    return True
                del placed_obj[obj_idx]

            print(f"[Final Failure] Could not place {obj['noun']} after {max_retry_per_node} retries and fallback.")
            return False

        success = dfs_traverse(tree)
        if not success:
            print("[Layout] Failed to generate valid layout.")
            return False, {}, 512, 512

        for idx, pos in placed_obj.items():
            cx, cy = pos["cxcy"]
            w, h = objects[idx]["w"], objects[idx]["h"]
            pos["x1y1"] = (cx - w / 2, cy - h / 2)
            pos["x2y2"] = (cx + w / 2, cy + h / 2)

        all_x = [pos["x1y1"][0] for pos in placed_obj.values()] + [pos["x2y2"][0] for pos in placed_obj.values()]
        all_y = [pos["x1y1"][1] for pos in placed_obj.values()] + [pos["x2y2"][1] for pos in placed_obj.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        margin_w = random.randint(80, 200)
        margin_h = random.randint(80, 200)
        canvas_width = int(max_x - min_x + margin_w)
        canvas_height = int(max_y - min_y + margin_h)
        offset_x = min_x - margin_w // 2
        offset_y = min_y - margin_h // 2

        for idx, pos in placed_obj.items():
            cx, cy = pos["cxcy"]
            pos["cxcy"] = (cx - offset_x, cy - offset_y)
            pos["x1y1"] = (pos["x1y1"][0] - offset_x, pos["x1y1"][1] - offset_y)
            pos["x2y2"] = (pos["x2y2"][0] - offset_x, pos["x2y2"][1] - offset_y)

        return True, placed_obj, canvas_width, canvas_height
            
            
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
        dataset.print_tree_structure(trees, idx_to_noun = {k: (k, v) for k, v in idx_to_noun.items()})
        referring_text = dataset.generate_referring_text(
            dataset.get_path_to_node(trees, data[1]['idx']), idx_to_noun
        )
        print(f"Referring expression: {referring_text}")
        status, placed_obj, width, height = dataset.optimize_layout(trees, data)
        print(f"Canvas size: {width} x {height}")
        visualize_layout(
                placed_obj=placed_obj,
                objects=data,
                tree=trees,
                idx_to_noun=idx_to_noun,
                canvas_width=width,
                canvas_height=height,
                output_path=f"visualizations/synthetics/layout_{i}.png"
            )
        print(placed_obj)
        print()
        print("-" * 50)
        
        
