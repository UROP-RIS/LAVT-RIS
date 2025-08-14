from data.synthetic.dataset_synthetic import SynthesisDataset
from PIL import Image
import numpy as np
import cv2
import torch
import math

class PolarOrdinalDataset(SynthesisDataset):
    
    def __init__(self, prob: float, root: str, dataset: str, split: str, max_tokens: int = 20, range_num: tuple = (4, 12), **kwargs):
        super().__init__(prob, root, dataset, split, max_tokens, **kwargs)
        self.range_num = range_num  # (min, max) total number of instances
        self.bg_color = (128, 128, 128)  # gray background
    
        # 数字到单词的映射
        self.number_words = {
            0: "twelve", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"
        }
        
        # 时钟位置的多样化表达
        self.clock_expressions = {
            0: ["12 o'clock", "twelve o'clock", "twelve o clock", "12 O'clock", "twelve O'clock", "twelve O clock"],
            1: ["1 o'clock", "one o'clock", "one o clock", "1 O'clock", "one O'clock", "one O clock"],
            2: ["2 o'clock", "two o'clock", "two o clock", "2 O'clock", "two O'clock", "two O clock"],
            3: ["3 o'clock", "three o'clock", "three o clock", "3 O'clock", "three O'clock", "three O clock"],
            4: ["4 o'clock", "four o'clock", "four o clock", "4 O'clock", "four O'clock", "four O clock"],
            5: ["5 o'clock", "five o'clock", "five o clock", "5 O'clock", "five O'clock", "five O clock"],
            6: ["6 o'clock", "six o'clock", "six o clock", "6 O'clock", "six O'clock", "six O clock"],
            7: ["7 o'clock", "seven o'clock", "seven o clock", "7 O'clock", "seven O'clock", "seven O clock"],
            8: ["8 o'clock", "eight o'clock", "eight o clock", "8 O'clock", "eight O'clock", "eight O clock"],
            9: ["9 o'clock", "nine o'clock", "nine o clock", "9 O'clock", "nine O'clock", "nine O clock"],
            10: ["10 o'clock", "ten o'clock", "ten o clock", "10 O'clock", "ten O'clock", "ten O clock"],
            11: ["11 o'clock", "eleven o'clock", "eleven o clock", "11 O'clock", "eleven O'clock", "eleven O clock"]
        }

    def get_diverse_clock_text(self, clock_position: int) -> str:
        """获取多样化的时钟位置表达"""
        expressions = self.clock_expressions[clock_position]
        
        # 基础表达：各种 o'clock 形式
        basic_expressions = expressions.copy()
        
        # 简化表达：at + 数字/单词
        number = clock_position if clock_position != 0 else 12
        word = self.number_words[clock_position]
        
        simple_expressions = [
            f"at {number}",
            f"at {word}",
        ]
        
        # 所有表达合并，给基础表达更高权重
        all_expressions = basic_expressions + simple_expressions
        
        return np.random.choice(all_expressions)
        
    def load_until_success(self) -> dict:
        idx = np.random.randint(0, len(self.index))
        data = self.load(idx)
        return data
    
    def add_padding(self, img: np.ndarray, target_aspect: float, pad_value: int = 128) -> np.ndarray:
        """
        对图像添加 padding，保持宽高比，支持单通道和三通道图像
        """
        is_gray = (len(img.shape) == 2)
        if is_gray:
            h, w = img.shape
            img_hwc = np.stack([img] * 3, axis=-1)  # 转为 (H, W, 3) 方便处理
        else:
            h, w = img.shape[:2]
            img_hwc = img
        current_aspect = w / h
        if current_aspect < target_aspect:
            new_w = int(h * target_aspect)
            new_h = h
            left = (new_w - w) // 2
            right = new_w - w - left
            top = bottom = 0
        else:
            new_h = int(w / target_aspect)
            new_w = w
            top = (new_h - h) // 2
            bottom = new_h - h - top
            left = right = 0
        padded = np.full((new_h, new_w, 3), pad_value, dtype=img_hwc.dtype)
        padded[top:top+h, left:left+w] = img_hwc
        if is_gray:
            padded = padded[:, :, 0]  # (H, W)
        return padded
    
    def calculate_polar_positions(self, num_instances: int, center_x: int, center_y: int, radius: int, patch_size: tuple):
        """
        计算极坐标排列的位置 - 支持随机分布在12个时钟位置，每个instance有随机大小
        
        Args:
            num_instances: 实例数量 (1-12)
            center_x, center_y: 圆心坐标
            radius: 半径
            patch_size: (patch_h, patch_w) patch的尺寸
            
        Returns:
            positions: list of dict containing position info
        """
        positions = []
        patch_h, patch_w = patch_size
        
        # 所有可能的时钟位置 (0-11 对应 12, 1, 2, ..., 11点)
        all_clock_positions = list(range(12))
        
        # 随机选择 num_instances 个位置，确保不重复
        selected_positions = np.random.choice(all_clock_positions, size=num_instances, replace=False)
        
        for i, clock_pos in enumerate(selected_positions):
            # 根据时钟位置计算角度
            # 12点钟为0度，1点钟为30度，顺时针方向
            angle = clock_pos * (2 * math.pi / 12) - math.pi/2  # -π/2 让0度指向12点
            
            # 为每个instance生成随机缩放比例 (0.75 - 1.5倍)
            scale_factor = np.random.uniform(0.75, 1.5)
            scaled_h = int(patch_h * scale_factor)
            scaled_w = int(patch_w * scale_factor)
            
            # 计算实际坐标 (注意Y轴向下为正)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # 确保patch不会超出边界，调整位置
            x = max(scaled_w//2, min(x, center_x*2 - scaled_w//2))
            y = max(scaled_h//2, min(y, center_y*2 - scaled_h//2))
            
            # 转换为左上角坐标
            x1 = int(x - scaled_w//2)
            y1 = int(y - scaled_h//2)
            
            positions.append({
                "index": i,
                "clock_position": clock_pos,
                "position": (x1, y1),
                "angle": angle,
                "center_position": (int(x), int(y)),
                "scale_factor": scale_factor,
                "scaled_size": (scaled_h, scaled_w)
            })
        
        return positions

    def scale_patch_and_mask(self, patch: np.ndarray, mask: np.ndarray, scale_factor: float):
        """
        按比例缩放patch和mask
        
        Args:
            patch: 原始patch (H, W, 3)
            mask: 原始mask (H, W)
            scale_factor: 缩放比例
            
        Returns:
            scaled_patch: 缩放后的patch
            scaled_mask: 缩放后的mask
        """
        h, w = patch.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        if new_h <= 0 or new_w <= 0:
            return patch, mask
        
        # 缩放patch (使用双线性插值)
        scaled_patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 缩放mask (使用最近邻插值保持二值性)
        scaled_mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return scaled_patch, scaled_mask

    def paste(self, bg: np.ndarray, patch: np.ndarray, mask: np.ndarray, x: int, y: int, scale_factor: float = 1.0):
        """
        将patch粘贴到背景图像上的指定位置，支持缩放
        
        Args:
            bg: 背景图像 (H, W, 3)
            patch: 要粘贴的patch (H, W, 3)
            mask: patch的mask (H, W)
            x, y: 粘贴位置的左上角坐标
            scale_factor: 缩放比例
            
        Returns:
            bg: 修改后的背景图像
            full_obj_mask: 完整图像上该object的mask
        """
        # 先缩放patch和mask
        if scale_factor != 1.0:
            patch, mask = self.scale_patch_and_mask(patch, mask, scale_factor)
        
        patch_h, patch_w = patch.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        
        # 计算实际粘贴区域（防止越界）
        x_end = min(x + patch_w, bg_w)
        y_end = min(y + patch_h, bg_h)
        x_start = max(x, 0)
        y_start = max(y, 0)
        
        # 计算patch中对应的区域
        patch_x_start = max(-x, 0)
        patch_y_start = max(-y, 0)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_end = patch_y_start + (y_end - y_start)
        
        if x_start >= x_end or y_start >= y_end or patch_x_start >= patch_w or patch_y_start >= patch_h:
            # 没有有效的粘贴区域
            return bg, np.zeros((bg_h, bg_w), dtype=np.uint8)
        
        # 提取要粘贴的区域
        patch_crop = patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        mask_crop = mask[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        
        # 创建3通道的mask用于图像粘贴
        mask_3d = mask_crop[:, :, np.newaxis].astype(bool)
        
        # 粘贴patch到背景
        bg[y_start:y_end, x_start:x_end] = np.where(
            mask_3d,
            patch_crop,
            bg[y_start:y_end, x_start:x_end]
        )
        
        # 创建完整的object mask
        full_obj_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        full_obj_mask[y_start:y_end, x_start:x_end] = mask_crop
        
        return bg, full_obj_mask

    def __call__(self, idx=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx is None:
            idx = np.random.randint(0, len(self.index))
        
        data = self.load_until_success()
        img_array = data['img']
        noun = data['noun'] 
        mask = data['mask']
        patch_mask, patch = self._crop_mask_and_patch(mask, img_array)
        
        # 随机确定实例数量 (1-12个)
        num_instances = np.random.randint(1, 13)  # 1到12个实例
        target_idx = np.random.randint(0, num_instances)  # 被指代的是第几个
        
        # 计算画布尺寸和圆形排列参数
        patch_h, patch_w = patch.shape[:2]
        
        # 考虑最大缩放比例来计算基础半径
        max_scale = 1.5
        base_radius = max(200, max(patch_h, patch_w) * max_scale * 1.5)  # 基础半径要考虑最大缩放
        
        # 根据实例数量微调半径，避免太拥挤或太稀疏
        if num_instances <= 3:
            radius = int(base_radius * 0.8)  # 少数实例时，稍微缩小半径
        elif num_instances >= 10:
            radius = int(base_radius * 1.2)  # 多数实例时，稍微扩大半径
        else:
            radius = int(base_radius)  # 确保是整数
        
        # 画布尺寸（确保能容纳整个圆形排列，考虑最大缩放）
        canvas_margin = max(patch_h, patch_w) * max_scale
        canvas_size = int((radius + canvas_margin) * 2)  # 确保是整数
        center_x = center_y = canvas_size // 2
        
        # 计算所有位置 - 现在会随机分布在12个时钟位置中，并包含缩放信息
        positions = self.calculate_polar_positions(
            num_instances, center_x, center_y, radius, (patch_h, patch_w)
        )
        
        # 创建背景
        use_real_bg = False
        if use_real_bg:
            # 使用真实背景
            bg_idx = np.random.randint(0, len(self.index))
            bg_data = self.load(bg_idx)
            bg = bg_data['img']  # (H, W, 3), uint8
            
            # 调整背景尺寸 - 确保canvas_size是整数
            bg = cv2.resize(bg, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
            
            # 模糊背景
            kernel_size = max(31, int(min(patch_h, patch_w) * 0.3) // 2 * 2 + 1)
            bg = cv2.blur(bg, (kernel_size, kernel_size))
            
            # 亮度匹配
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
            bg_brightness = bg_gray.mean()
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            patch_brightness = patch_gray.mean()
            brightness_ratio = bg_brightness / (patch_brightness + 1e-5)
            adjusted_patch = np.clip(patch.astype(np.float32) * brightness_ratio, 0, 255).astype(np.uint8)
            patch = adjusted_patch
        else:
            bg = np.full((canvas_size, canvas_size, 3), self.bg_color, dtype=np.uint8)
        
        full_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        # 粘贴所有instances，每个都有不同的缩放比例
        for i, pos_data in enumerate(positions):
            x, y = pos_data["position"]
            scale_factor = pos_data["scale_factor"]
            
            # 使用缩放比例粘贴
            bg, full_obj_mask = self.paste(bg, patch, patch_mask, x, y, scale_factor)
            
            if i == target_idx:
                full_mask = full_obj_mask
                target_clock_pos = pos_data["clock_position"]
        
        # 生成多样化的时钟位置文本
        clock_text = self.get_diverse_clock_text(target_clock_pos)
        
        # 根据实例数量调整描述策略
        templates = []
        
        # 基础时钟位置描述 - 现在使用多样化的时钟文本
        # 判断是否是简化表达 (at xxx)
        is_simple_at = clock_text.startswith("at ")
        
        if is_simple_at:
            # 对于 "at xxx" 形式，使用不同的模板
            templates.extend([
                f"{noun} {clock_text}",
                f"the {noun} {clock_text}",
                f"{noun} positioned {clock_text}",
                f"the {noun} positioned {clock_text}",
                f"{noun} located {clock_text}",
                f"the {noun} located {clock_text}",
            ])
        else:
            # 对于 "xxx o'clock" 形式，使用传统模板
            templates.extend([
                f"{noun} at {clock_text}",
                f"the {noun} at {clock_text}",
                f"{noun} on {clock_text}",
                f"the {noun} on {clock_text}",
            ])
        
        # 如果只有少数几个实例，增加更多样化的描述
        if num_instances <= 4:
            if is_simple_at:
                templates.extend([
                    f"find the {noun} {clock_text}",
                    f"look for the {noun} {clock_text}",
                    f"select the {noun} {clock_text}",
                    f"the {noun} that is {clock_text}",
                ])
            else:
                templates.extend([
                    f"{noun} positioned at {clock_text}",
                    f"the {noun} positioned at {clock_text}",
                    f"{noun} located at {clock_text}",
                    f"the {noun} located at {clock_text}",
                    f"find the {noun} at {clock_text}",
                    f"look for the {noun} at {clock_text}",
                    f"select the {noun} at {clock_text}",
                    f"the {noun} that is at {clock_text}",
                ])
        
        # 特殊位置的额外描述 - 只有在实例较少时才使用方向描述
        if num_instances <= 4:
            if target_clock_pos == 0:  # 12点
                templates.extend([
                    f"the {noun} at the top",
                    f"the {noun} on top",
                    f"the topmost {noun}",
                ])
            elif target_clock_pos == 6:  # 6点
                templates.extend([
                    f"the {noun} at the bottom",
                    f"the {noun} on bottom", 
                    f"the bottommost {noun}",
                ])
            elif target_clock_pos == 3:  # 3点
                templates.extend([
                    f"the {noun} on the right",
                    f"the {noun} to the right",
                    f"the rightmost {noun}",
                ])
            elif target_clock_pos == 9:  # 9点
                templates.extend([
                    f"the {noun} on the left",
                    f"the {noun} to the left",
                    f"the leftmost {noun}",
                ])
        
        # 只有实例很少时才使用区域描述，避免歧义
        if num_instances <= 2:
            if target_clock_pos in [11, 0, 1]:  # 上方区域
                templates.extend([
                    f"the {noun} in the upper area",
                    f"the {noun} towards the top",
                ])
            elif target_clock_pos in [5, 6, 7]:  # 下方区域
                templates.extend([
                    f"the {noun} in the lower area", 
                    f"the {noun} towards the bottom",
                ])
            elif target_clock_pos in [2, 3, 4]:  # 右方区域
                templates.extend([
                    f"the {noun} on the right side",
                    f"the {noun} towards the right",
                ])
            elif target_clock_pos in [8, 9, 10]:  # 左方区域
                templates.extend([
                    f"the {noun} on the left side",
                    f"the {noun} towards the left",
                ])
        
        # 加权选择文本 - 随着实例数量增加，更偏向精确的时钟位置描述
        weights = []
        for template in templates:
            weight = 1.0
            
            # 时钟位置描述的权重随实例数量增加
            # 检查是否包含时钟相关词汇
            has_clock_ref = any(word in template.lower() for word in ["o'clock", "o clock", "at twelve", "at one", "at two", "at three", "at four", "at five", "at six", "at seven", "at eight", "at nine", "at ten", "at eleven"])
            
            if has_clock_ref:
                if num_instances >= 8:
                    weight *= 4.0  # 实例很多时，强烈偏向时钟描述
                elif num_instances >= 5:
                    weight *= 3.0
                else:
                    weight *= 2.0
            
            # 精确位置描述
            if "at " in template or "on " in template:
                weight *= 1.5
                
            # 方向描述在实例多时权重降低
            if any(word in template for word in ["top", "bottom", "left", "right", "upper", "lower"]) and not has_clock_ref:
                if num_instances >= 8:
                    weight *= 0.3  # 实例很多时，降低方向描述权重
                elif num_instances >= 5:
                    weight *= 0.6
                    
            weights.append(weight)
        
        weights = np.array(weights)
        text = np.random.choice(templates, p=weights / weights.sum())
        
        # 添加padding使图像为正方形
        mean_value = int(np.mean(bg))
        bg = self.add_padding(bg, target_aspect=1.0, pad_value=mean_value)
        full_mask = self.add_padding(full_mask, target_aspect=1.0, pad_value=0)
        
        # 最终处理
        if not self.load_raw_data:
            full_mask_img = Image.fromarray(full_mask.astype(np.uint8)).convert("P")
            img_pil = Image.fromarray(bg.astype(np.uint8)).convert("RGB")
            img_tensor, mask_tensor = self.apply_transforms(img_pil, full_mask_img)
            input_ids, attention_mask = self.tokenize_text(text)
            return img_tensor, mask_tensor, input_ids, attention_mask
        else:
            return bg, text, full_mask


if __name__ == "__main__":
    import os
    import cv2
    import numpy as np

    os.makedirs("visualizations/synthetics", exist_ok=True)

    dataset = PolarOrdinalDataset(
        prob=1.0,
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc", 
        split="train", 
        max_tokens=20, 
        range_num=(6, 10), 
        load_raw_data=True
    )

    print("== Testing PolarOrdinalDataset (clock-position referring) ==")

    for i in range(20):
        result = dataset()
        if result is None:
            print(f"[{i+1}] Failed to generate sample.")
            continue

        img, txt, mask = result
        print(f"Referring text: {txt}")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")
        
        # 准备可视化图像
        vis_img = img.copy()
        if vis_img.max() <= 1.0:
            vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # 创建target mask的彩色版本 (绿色)
        target_mask = (mask > 0).astype(np.uint8) * 255
        target_mask_colored = np.zeros_like(vis_img)
        target_mask_colored[:, :, 1] = target_mask  # Green channel

        # 叠加
        overlay = cv2.addWeighted(vis_img, 0.6, target_mask_colored, 0.4, 0)

        # 添加文字
        font_scale = 0.6 * vis_img.shape[1] / 640
        cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        # 保存
        save_path = f"visualizations/synthetics/polar_example_{i+1}.png"
        cv2.imwrite(save_path, overlay)
        print(f"Saved to {save_path}\n")