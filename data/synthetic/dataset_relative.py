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

RELATIONS = {
    "left": {
        "templates": [
            "{A} is to the left of {B}",
            "the {A} located to the left of the {B}",
            "the {A} positioned on the left side of the {B}",
            "the {A} seen on the left relative to the {B}",
            "the {A} that is on the left when facing the {B}",
            "the {A} situated to the left of the {B}",
            "the {A} standing to the left of the {B}",
            "the {A} placed to the left of the {B}",
            "the {A} aligned to the left of the {B}",
            "the {A} that appears to the left of the {B}"
        ]
    },
    "right": {
        "templates": [
            "{A} is to the right of {B}",
            "the {A} located to the right of the {B}",
            "the {A} positioned on the right side of the {B}",
            "the {A} seen on the right relative to the {B}",
            "the {A} that is on the right when facing the {B}",
            "the {A} situated to the right of the {B}",
            "the {A} standing to the right of the {B}",
            "the {A} placed to the right of the {B}",
            "the {A} aligned to the right of the {B}",
            "the {A} that appears to the right of the {B}"
        ]
    },
    "above": {
        "templates": [
            "{A} is above the {B}",
            "the {A} located directly above the {B}",
            "the {A} positioned over the {B}",
            "the {A} floating above the {B}",
            "the {A} suspended above the {B}",
            "the {A} sitting on top of the {B}",
            "the {A} resting on the upper part of the {B}",
            "the {A} placed at the top of the {B}",
            "the {A} that is higher than the {B}",
            "the {A} seen above the {B} in the scene"
        ]
    },
    "below": {
        "templates": [
            "{A} is below the {B}",
            "the {A} located directly below the {B}",
            "the {A} positioned under the {B}",
            "the {A} beneath the {B}",
            "the {A} underneath the {B}",
            "the {A} sitting at the bottom of the {B}",
            "the {A} resting under the {B}",
            "the {A} placed beneath the {B}",
            "the {A} that is lower than the {B}",
            "the {A} seen below the {B} in the scene"
        ]
    },
    "behind": {
        "templates": [
            "{A} is behind the {B}",
            "the {A} located at the back of the {B}",
            "the {A} positioned behind the {B}",
            "the {A} hidden behind the {B}",
            "the {A} obscured by the {B}",
            "the {A} situated at the rear of the {B}",
            "the {A} that is farther from the viewer than the {B}",
            "the {A} seen behind the {B} from this viewpoint",
            "the {A} that lies in the shadow of the {B}",
            "the {A} not visible because it's behind the {B}"
        ]
    },
    "in_front_of": {
        "templates": [
            "{A} is in front of the {B}",
            "the {A} located in front of the {B}",
            "the {A} positioned ahead of the {B}",
            "the {A} blocking the view of the {B}",
            "the {A} closer to the viewer than the {B}",
            "the {A} seen in front of the {B} from this angle",
            "the {A} standing in front of the {B}",
            "the {A} that appears before the {B}",
            "the {A} partially occluding the {B}",
            "the {A} that is in the foreground relative to the {B}"
        ]
    },
    "near": {
        "templates": [
            "{A} is near the {B}",
            "the {A} located close to the {B}",
            "the {A} in the vicinity of the {B}",
            "the {A} in proximity to the {B}",
            "the {A} adjacent to the {B}",
            "the {A} beside the {B}",
            "the {A} nearby the {B}",
            "the {A} within a short distance from the {B}",
            "the {A} that is spatially close to the {B}",
            "the {A} that appears near the {B} in the image"
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
    
    def load_objects(self, num_distinc_objs: int, copy_num = 0) -> list:
        """
        Load a list of objects from the noun dictionary.
        """
        objects = []
        for idx in range(num_distinc_objs):
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
        if copy_num > 0:
            copied = random.sample(objects, 1)[0]
            for _ in range(copy_num):
                new_obj = copied.copy()
                new_obj["idx"] = len(objects)
                objects.append(new_obj)
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
        # print(f"{prefix}{connector}{noun_str}{rel_str}")
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
                    left_edge_min = right_edge_max - 1.0 * max(pw, cw)
                    cx = random.uniform(left_edge_min + cw / 2, right_edge_max - cw / 2)
                    cy = random.uniform(parent_cy - ph * 0.3, parent_cy + ph * 0.3)
                elif relation == "right":
                    left_edge_min = parent_cx + pw / 2 + min_gap
                    right_edge_max = left_edge_min + 1.0 * max(pw, cw)
                    cx = random.uniform(left_edge_min + cw / 2, right_edge_max - cw / 2)
                    cy = random.uniform(parent_cy - ph * 0.3, parent_cy + ph * 0.3)
                elif relation == "above":
                    bottom_edge_max = parent_cy - ph / 2 - min_gap
                    top_edge_min = bottom_edge_max - 1.0 * max(ph, ch)
                    cy = random.uniform(top_edge_min + ch / 2, bottom_edge_max - ch / 2)
                    cx = random.uniform(parent_cx - pw * 0.3, parent_cx + pw * 0.3)
                elif relation == "below":
                    top_edge_min = parent_cy + ph / 2 + min_gap
                    bottom_edge_max = top_edge_min + 1.0 * max(ph, ch)
                    cy = random.uniform(top_edge_min + ch / 2, bottom_edge_max - ch / 2)
                    cx = random.uniform(parent_cx - pw * 0.3, parent_cx + pw * 0.3)
                elif relation == "near":
                    radius_max = 0.2 * math.sqrt(pw**2 + ph**2)
                    radius_min = 0.2 * radius_max
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(radius_min, radius_max)
                    cx = parent_cx + dist * math.cos(angle)
                    cy = parent_cy + dist * math.sin(angle)
                elif relation == "in_front_of":
                    offset = random.uniform(0, 30)
                    cx = parent_cx + random.uniform(-15, 15)
                    cy = parent_cy - offset
                elif relation == "behind":
                    offset = random.uniform(0, 30)
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
                # print(f"[Root] Failed to place root {obj['noun']} after {max_retry_root} retries.")
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
                    # print(f"[Retry] Placed but children failed: {obj['noun']} at ({x}, {y}), attempt {retry + 1}")
                # else:
                    # print(f"[Retry] Failed to place {obj['noun']} with '{relation}' (attempt {retry + 1})")

            if last_best:
                x, y, max_iou = last_best
                placed_obj[obj_idx] = {"cxcy": (x, y)}
                # print(f"[Fallback] Using best candidate for {obj['noun']} at ({x}, {y}) with IoU={max_iou:.3f}")
                if dfs_traverse_children(node, obj):
                    return True
                del placed_obj[obj_idx]

            # print(f"[Final Failure] Could not place {obj['noun']} after {max_retry_per_node} retries and fallback.")
            return False

        success = dfs_traverse(tree)
        if not success:
            # print("[Layout] Failed to generate valid layout.")
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
    
    def compute_z_order(self, tree, placed_obj):
        """
        根据空间关系和树结构确定每个实例的绘制顺序（z-order），避免遮挡错误。
        返回：list of idx，按绘制顺序排列（先画的在前，后画的在后）

        规则：
        - "in_front_of": 前者在上层（后画）
        - "behind": 前者在下层（先画）
        - 默认：子节点在父节点之上（后画）
        - 冲突时优先级：显式空间关系 > 树结构
        """
        # 初始化默认顺序：按 idx 排序（确定性）
        nodes = list(placed_obj.keys())
        z_order = []

        # 显式空间关系优先级
        below_relations = []  # (under, over) 表示 under 应该先画
        explicit_pairs = set()

        def extract_relations(node):
            for child in node['children']:
                rel = child['relation']
                parent_idx = node['idx']
                child_idx = child['idx']
                pair = (min(parent_idx, child_idx), max(parent_idx, child_idx))

                if rel == "in_front_of":
                    # child 在 parent 前面 → child 后画（在上层）
                    below_relations.append((parent_idx, child_idx))
                    explicit_pairs.add(pair)
                elif rel == "behind":
                    # child 在 parent 后面 → child 先画（在下层）
                    below_relations.append((child_idx, parent_idx))
                    explicit_pairs.add(pair)
                extract_relations(child)

        extract_relations(tree)

        # 构建依赖图
        from collections import defaultdict, deque
        graph = defaultdict(list)
        indegree = {n: 0 for n in nodes}

        for u, v in below_relations:
            if u not in nodes or v not in nodes:
                continue
            graph[u].append(v)
            indegree[v] = indegree.get(v, 0) + 1

        # 添加默认父子顺序：parent 先画，child 后画
        def add_parent_child_order(node):
            for child in node['children']:
                parent_idx = node['idx']
                child_idx = child['idx']
                pair = (min(parent_idx, child_idx), max(parent_idx, child_idx))
                if pair not in explicit_pairs:  # 无显式关系才加默认
                    if child_idx not in graph[parent_idx]:
                        graph[parent_idx].append(child_idx)
                        indegree[child_idx] = indegree.get(child_idx, 0) + 1
                add_parent_child_order(child)
        add_parent_child_order(tree)

        # 拓扑排序（Kahn 算法）
        queue = deque([n for n in nodes if indegree[n] == 0])
        while queue:
            u = queue.popleft()
            z_order.append(u)
            for v in graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        # 安全兜底
        if len(z_order) != len(nodes):
            missing = set(nodes) - set(z_order)
            z_order.extend(missing)

        return z_order

    def paste_instance_by_order(self, canvas_width, canvas_height, placed_obj, z_order, refer_idx, objects):
        """
        按 z-order 将实例粘贴到画布上，并生成符合遮挡关系的 referring instance mask。
        
        Args:
            canvas_width: 画布宽度
            canvas_height: 画布高度
            placed_obj: 布局字典，idx -> { "cxcy": (x, y) }
            z_order: 绘制顺序（从底层到顶层）
            refer_idx: 指代对象的 idx
            objects: 对象列表，dict with 'img', 'mask', 'w', 'h', 'idx'

        Returns:
            canvas: 合成图像 (H, W, 3)
            final_mask: referring instance 的可见部分 mask (H, W)
        """
        # canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 128  # 灰色背景
        bg, bg_color = self.create_scrambled_background_from_single_image(None, rows = 16, cols = 16, blur_kernel_ratio=0.04)
        canvas = cv2.resize(bg, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
        final_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)  # 指代对象的初始 mask
        idx_to_obj = {obj['idx']: obj for obj in objects}  # 快速查找

        # 第一遍：按 z-order 贴图，并记录 refer_idx 的 mask
        for idx in z_order:
            obj = idx_to_obj[idx]
            pos = placed_obj[idx]
            cx, cy = pos["cxcy"]
            x = int(cx - obj['w'] // 2)
            y = int(cy - obj['h'] // 2)

            # 贴图
            canvas, _ = self.paste(canvas, obj['img'], obj['mask'], x, y)

            # 记录 refer_idx 的完整 mask（未考虑遮挡）
            if idx == refer_idx:
                _, temp_mask = self.paste(np.zeros_like(canvas), obj['img'], obj['mask'], x, y)
                final_mask = temp_mask.copy()

        # 第二步：生成遮挡 mask（所有在 refer_idx 之后绘制的对象）
        occluder_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        past_refer = False

        for idx in z_order:
            if idx == refer_idx:
                past_refer = True
                continue
            if past_refer:  # 这些对象在 refer_idx 之上，会遮挡它
                obj = idx_to_obj[idx]
                pos = placed_obj[idx]
                cx, cy = pos["cxcy"]
                x = int(cx - obj['w'] // 2)
                y = int(cy - obj['h'] // 2)
                _, temp_mask = self.paste(np.zeros_like(canvas), obj['img'], obj['mask'], x, y)
                occluder_mask = np.maximum(occluder_mask, temp_mask)

        # 最终 mask：refer_idx 可见部分 = 原始 mask - 被上层遮挡区域
        final_mask = np.where(occluder_mask > 0, 0, final_mask).astype(np.uint8)

        return canvas, final_mask
    
    def __call__(self, idx=None):
        """
        生成一个合成图像和对应的 referring expression。
        """
        if idx is None:
            idx = np.random.randint(0, len(self.index))
        
        objects = self.load_objects(num_distinc_objs=4, copy_num=random.randint(0, 2))
        trees = self.build_tree_bfs(objects)
        idx_to_noun = {obj['idx']: obj['noun'] for obj in objects}
        
        ## Filter out the root
        chosen = random.choice(list(set(range(len(objects))) - set([trees["idx"]])))

        referring_text = self.generate_referring_text(
            self.get_path_to_node(trees, objects[chosen]['idx']), idx_to_noun
        )
        status, placed_obj, width, height = self.optimize_layout(trees, objects)
        z_order = self.compute_z_order(trees, placed_obj)
        canvas, final_mask = self.paste_instance_by_order(
            width, height, placed_obj, z_order, objects[chosen]['idx'], objects
        )
        
        padded_canvas = self.add_padding(canvas, target_aspect=1.0, pad_value=128)
        padded_mask = self.add_padding(final_mask, target_aspect=1.0, pad_value=0)
        
               # 最终处理
        if not self.load_raw_data:
            full_mask_img = Image.fromarray(padded_mask.astype(np.uint8)).convert("P")
            img_pil = Image.fromarray(padded_canvas.astype(np.uint8)).convert("RGB")
            img_tensor, mask_tensor = self.apply_transforms(img_pil, full_mask_img)
            input_ids, attention_mask = self.tokenize_text(referring_text)
            return img_tensor, mask_tensor, input_ids, attention_mask
        else:
            return padded_canvas, padded_mask, referring_text


        
        
    

            
if __name__ == "__main__":
    dataset = RelativeDataset(
        prob=0.5, 
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc", 
        split="train", 
        max_tokens=20, 
        load_raw_data=True
    )
    
    for i in range(50):
        canvas, mask, text = dataset()
        print(f"Generated text: {text}")
        print(f"Canvas shape: {canvas.shape}, Mask shape: {mask.shape}")
        vis_img = dataset.get_vis_img(
            canvas, mask, text
        )
        print(vis_img.shape)
        cv2.imwrite(f"visualizations/synthetics/relative_{i}.png", vis_img)
        
   
        
        
