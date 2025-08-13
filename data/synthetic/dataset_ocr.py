from data.synthetic.dataset_synthetic import SynthesisDataset
from data.synthetic.utils.number import NumberGenerator
import math
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from PIL import Image

class NumberOcrDataset(SynthesisDataset):
    def __init__(self, prob: float, root: str, dataset: str, split: str, max_tokens: int = 20, range_num: tuple = (1, 4), number_range: tuple = (1, 101), other_class_odd: float = 0.3, **kwargs):
        """
        range_num: min, max num of instances in the image
        number_range: min, max value of numbers
        other_class_odd: probability of having other class instances
        """
        super().__init__(prob, root, dataset, split, max_tokens, **kwargs)
        self.range_num = range_num
        self.bg_color = (128, 128, 128)
        self.other_class_odd = float(np.clip(other_class_odd, 0, 1))
        self.number_range = number_range  # (min, max) range of numbers to generate
    
    def get_conditions(self):
        # random gen a int in range [range_num[0], range_num[1]]
        num_instances = np.random.randint(self.range_num[0], self.range_num[1] + 1)
        number_list = [np.random.randint(self.number_range[0], self.number_range[1] + 1) for _ in range(num_instances)]
        target_number = number_list[0]  # 只指代第一个数字
        for i in range(1, len(number_list)):
            if number_list[i] == target_number:
                number_list[i] += 1 # make sure that the number as target is unique
        other_class = True if np.random.rand() < self.other_class_odd else False
        return {
            "num_instances": num_instances,
            "number_list": number_list,
            "target_number": target_number,
            "other_class": other_class
        }
    
    @staticmethod
    def get_number_mask(number_list: list):
        number_generator_list = [NumberGenerator(num) for num in number_list]
        [number_generator_list[i].aug() for i in range(len(number_generator_list))]
        return [[ng.image, ng.mask] for ng in number_generator_list]

    def load_data(self, idx, other_class = False):
        if idx is None:
            idx = np.random.randint(0, len(self.index))
        data = self.load(idx)
        img_array = data['img']
        noun = data['txt'] 
        mask = data['mask']
        other_img = None
        other_mask = None
        other_txt = None
        if other_class:
            other_idx = np.random.randint(0, len(self.index))
            other_data = self.load(other_idx)
            other_img = other_data['img']
            other_mask = other_data['mask']
            other_txt = other_data["txt"]
        return {
            "this_data": {
                "img": img_array,
                "txt": noun,
                "mask": mask
            },
            "other_data": {
                "img": other_img,
                "txt": other_txt,
                "mask": other_mask
            }
        }
    
    @staticmethod
    def number_to_words(num):
        # Check if number is in valid range
        assert 0 <= num <= 200, "Number out of range (0-200)"

        # Define word mappings
        ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
                "seventeen", "eighteen", "nineteen"]
        
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        # Handle special cases
        if num < 20:
            return ones[num]
        
        elif num < 100:
            if num % 10 == 0:
                return tens[num // 10]
            else:
                return tens[num // 10] + "-" + ones[num % 10]
        
        elif num < 200:
            if num % 100 == 0:
                return ones[num // 100] + " hundred"
            else:
                return ones[num // 100] + " hundred " + NumberOcrDataset.number_to_words(num % 100)

        else:  # num == 200
            return "two hundred"
    
    @staticmethod
    def generate_txt(noun: str, number: int):
        list_of_template = [
            f"{noun} marked {number}",
            f"{number} {noun}",
            str(number),
            f"{noun} marked {NumberOcrDataset.number_to_words(number)}",
            f"{NumberOcrDataset.number_to_words(number)} {noun}",
            NumberOcrDataset.number_to_words(number),
            f"{noun} {number}",
            f"{noun} {NumberOcrDataset.number_to_words(number)}",
            f"{noun} with {number}",
            f"{noun} with {NumberOcrDataset.number_to_words(number)}"
        ]
        human_word_list = [
                            'man', 'male', 'player', 'batter', 'catcher', 
                            'umpire', 'child', 'boy', 'girl', 'person', 
                            'woman', 'female', 'lady', 'guy'
                        ]
        is_human = True if any(word in noun.lower() for word in human_word_list) else False
        if is_human:
            human_template = [
                f"{noun} wearing {NumberOcrDataset.number_to_words(number)}",
                f"{NumberOcrDataset.number_to_words(number)} {noun} wearing",
                f"{noun} wearing {number}",
                f"{noun} in {NumberOcrDataset.number_to_words(number)}",
                f"{noun} in {number}",
            ]
            list_of_template.extend(human_template)
        sample_id = np.random.randint(0, len(list_of_template))
        return list_of_template[sample_id]
    
    @staticmethod
    def _crop_mask_and_patch(mask: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop the mask and patch to the bounding box of the mask.
        If the cropped result's minimum dimension is less than 256, scale proportionally 
        to make the shortest side equal to 256.
        """
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None, None
        
        y1, y2, x1, x2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
        patch_cropped = patch[y1:y2+1, x1:x2+1]  # HxWx3
        mask_cropped = mask[y1:y2+1, x1:x2+1]    # HxW
        
        # 检查是否需要缩放
        h, w = mask_cropped.shape
        min_dim = min(h, w)
        
        if min_dim < 256:
            # 计算缩放比例，使最短边变为256
            scale_factor = 256.0 / min_dim
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            # 使用cv2进行等比缩放
            patch_resized = cv2.resize(patch_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_cropped.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            return mask_resized, patch_resized
        
        return mask_cropped, patch_cropped
    
    @staticmethod
    def find_best_fit_rect(mask: torch.Tensor, h: int, w: int):
        """
        找到原始矩形 (h, w) 等比例缩放后，能完全放入 mask==1 区域的最大版本。
        
        Args:
            mask: shape (1, H, W), dtype=torch.bool or 0/1 int/float
            h, w: 原始矩形的高度和宽度
        
        Returns:
            top, left: 最佳左上角位置
            h_prime, w_prime: 缩放后的高度和宽度
            scale: 缩放比例
        """
        device = mask.device
        mask = mask.squeeze(0)  # (H, W)
        H, W = mask.shape

        # 转为 bool
        if mask.dtype != torch.bool:
            mask = mask > 0.5

        # 如果原始矩形比整个 mask 还大，先缩放到能放下的最大比例
        max_scale = min(H / h, W / w)
        aspect_ratio = w / h  # 宽高比

        # 二分查找最大可行 scale
        lo, hi = 0.0, max_scale
        best_scale = 0.0
        best_pos = (0, 0)
        best_size = (0, 0)

        # 为了加速，我们只检查整数尺寸的矩形
        # 我们最多检查几百个 scale，或者用二分迭代
        # 这里用二分法，迭代 50 次足够
        tol = 1e-4
        iter_count = 0
        while hi - lo > tol and iter_count < 100:
            mid = (lo + hi) / 2
            h_s = int(mid * h)
            w_s = int(mid * w)

            if h_s <= 0 or w_s <= 0:
                lo = mid
                iter_count += 1
                continue

            # 检查是否存在位置 (i, j)，使得 [i:i+h_s, j:j+w_s] 全为 True
            found, top, left = NumberOcrDataset.can_place(mask, h_s, w_s)
            if found:
                best_scale = mid
                best_pos = (top, left)
                best_size = (h_s, w_s)
                lo = mid
            else:
                hi = mid
            iter_count += 1

        top, left = best_pos
        h_prime, w_prime = best_size

        return top, left, h_prime, w_prime

    @staticmethod
    def can_place(mask: torch.Tensor, h_s: int, w_s: int):
        """
        检查是否能在 mask 中放置一个 h_s x w_s 的矩形，完全在 True 区域内。
        返回 (found: bool, top: int, left: int)
        """
        H, W = mask.shape
        if h_s > H or w_s > W:
            return False, 0, 0

        # 使用 2D 卷积或滑动窗口求和
        # 创建一个全 1 的 kernel，卷积后值等于 h_s * w_s 表示全为 1
        kernel = torch.ones((1, 1, h_s, w_s), dtype=torch.float32, device=mask.device)
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # 使用卷积计算每个位置的和
        conv = torch.nn.functional.conv2d(mask_float, kernel, stride=1, padding=0)
        valid_area = conv[0, 0]  # (H - h_s + 1, W - w_s + 1)
        target = h_s * w_s

        # 找到所有等于 target 的位置
        match_map = (valid_area == target)
        if match_map.any():
            # 取第一个匹配的位置
            idx = match_map.nonzero(as_tuple=False)[0]
            top = idx[0].item()
            left = idx[1].item()
            return True, top, left

        return False, 0, 0
    
    @staticmethod
    def put_number_into_image(patch: torch.tensor, patch_mask: torch.tensor, number_img: torch.tensor, number_mask: torch.tensor):
        _, h, w = number_img.shape  # 获取数字图像的尺寸
        top, left, h_prime, w_prime = NumberOcrDataset.find_best_fit_rect(patch_mask, h, w)
        
        # resize the number_img, number_mask from h, w to h_prime, w_prime
        number_img = F.interpolate(number_img.unsqueeze(0), size=(h_prime, w_prime), mode="bilinear").squeeze(0)
        number_mask = F.interpolate(number_mask.unsqueeze(0), size=(h_prime, w_prime), mode="nearest").squeeze(0)
        
        # 创建patch的副本，避免修改原始数据
        modified_patch = patch.clone()
        
        # 获取patch的尺寸
        C, H, W = patch.shape
        
        # 确保粘贴区域不超出patch边界
        bottom = min(top + h_prime, H)
        right = min(left + w_prime, W)
        actual_h = bottom - top
        actual_w = right - left
        
        # 如果需要裁剪number_img和number_mask来适应patch边界
        if actual_h < h_prime or actual_w < w_prime:
            number_img = number_img[:, :actual_h, :actual_w]
            number_mask = number_mask[:, :actual_h, :actual_w]
        
        # 将number_mask扩展到3个通道以匹配RGB图像
        number_mask_rgb = number_mask.expand(3, -1, -1)  # (3, actual_h, actual_w)
        
        # 执行抠图粘贴：where number_mask==1 use number_img, else use original patch
        modified_patch[:, top:bottom, left:right] = torch.where(
            number_mask_rgb > 0.5,
            number_img,
            patch[:, top:bottom, left:right]
        )
        modified_patch[patch_mask.repeat(3,1,1) == 0] = 0
        return modified_patch
    
    def _save_patch_debug(self, patch_tensor, filename):
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs("visualizations/debug", exist_ok=True)
        
        patch_np = patch_tensor.permute(1, 2, 0).cpu().numpy()
        patch_np = (patch_np).astype(np.uint8)

        plt.figure(figsize=(6, 6))
        plt.imshow(patch_np)
        plt.axis('off')
        plt.title(filename)
        plt.savefig(f"visualizations/debug/{filename}", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Debug patch saved: visualizations/debug/{filename}")

    def _compose_image(self, instance_list, target_idx = 0):
        """
        将instance_list中的图像拼接成一个大图片
        
        Args:
            instance_list: [(image_tensor, mask_tensor), ...] 每个元素是(图像, mask)
            target_number: 目标数字，用于确定哪个instance是target
        
        Returns:
            big_image: 拼接后的大图像 (H, W, 3)
            full_mask: 只包含target instance的mask (H, W)
            target_idx: target instance在instance_list中的索引
        """
        num_instances = len(instance_list)
        
        # 计算网格布局
        cols = math.ceil(math.sqrt(num_instances))
        rows = math.ceil(num_instances / cols)
        
        # 计算每个instance的最大尺寸，用于统一cell大小
        max_h = max([img.shape[1] for img, _ in instance_list])  # img shape: (C, H, W)
        max_w = max([img.shape[2] for img, _ in instance_list])
        
        # 添加一些padding
        cell_h = max_h + 20
        cell_w = max_w + 20
        
        # 创建大背景图
        big_h = rows * cell_h
        big_w = cols * cell_w
        big_image = np.full((big_h, big_w, 3), self.bg_color, dtype=np.uint8)
        full_mask = np.zeros((big_h, big_w), dtype=np.uint8)
        
        # 确定target instance (第一个数字等于target_number的instance)
        target_idx = target_idx # 默认第一个
        
        # 随机排列instances的位置
        positions = [(i, j) for i in range(rows) for j in range(cols)][:num_instances]
        np.random.shuffle(positions)
        
        for idx, ((img_tensor, mask_tensor), (row, col)) in enumerate(zip(instance_list, positions)):
            # 转换tensor到numpy
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            mask_np = mask_tensor.squeeze(0).cpu().numpy()      # (H, W)
            
            # 确保数值范围正确
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            mask_np = (mask_np > 0.5).astype(np.uint8)
            
            # 计算在大图中的位置 (居中放置)
            start_y = row * cell_h + (cell_h - img_np.shape[0]) // 2
            start_x = col * cell_w + (cell_w - img_np.shape[1]) // 2
            end_y = start_y + img_np.shape[0]
            end_x = start_x + img_np.shape[1]
            
            # 确保不超出边界
            end_y = min(end_y, big_h)
            end_x = min(end_x, big_w)
            actual_h = end_y - start_y
            actual_w = end_x - start_x
            
            # 可能需要裁剪图像
            img_to_paste = img_np[:actual_h, :actual_w]
            mask_to_paste = mask_np[:actual_h, :actual_w]
            
            # 粘贴图像 (只在mask为1的地方)
            mask_3d = mask_to_paste[..., None]  # (H, W, 1)
            big_image[start_y:end_y, start_x:end_x] = np.where(
                mask_3d > 0,
                img_to_paste,
                big_image[start_y:end_y, start_x:end_x]
            )
            
            # 如果这是target instance，更新full_mask
            if idx == target_idx:
                full_mask[start_y:end_y, start_x:end_x] = mask_to_paste
        
        return big_image, full_mask, target_idx

    def _paste_instance(self, big_image, full_mask, img_tensor, mask_tensor, start_x, start_y, is_target=False):
        """
        将单个instance粘贴到大图像的指定位置
        
        Args:
            big_image: 大背景图像
            full_mask: 完整的mask (只有target instance为1)
            img_tensor: 要粘贴的图像 tensor (C, H, W)
            mask_tensor: 对应的mask tensor (1, H, W)
            start_x, start_y: 粘贴的起始位置
            is_target: 是否为target instance
        
        Returns:
            更新后的 big_image 和 full_mask
        """
        # 转换为numpy
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        mask_np = mask_tensor.squeeze(0).cpu().numpy()      # (H, W)
        
        # 确保数值范围
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        h, w = img_np.shape[:2]
        big_h, big_w = big_image.shape[:2]
        
        # 计算实际粘贴区域
        end_x = min(start_x + w, big_w)
        end_y = min(start_y + h, big_h)
        actual_w = end_x - start_x
        actual_h = end_y - start_y
        
        if actual_w <= 0 or actual_h <= 0:
            return big_image, full_mask
        
        # 裁剪要粘贴的内容
        img_crop = img_np[:actual_h, :actual_w]
        mask_crop = mask_np[:actual_h, :actual_w]
        
        # 粘贴图像
        mask_3d = mask_crop[..., None]
        big_image[start_y:end_y, start_x:end_x] = np.where(
            mask_3d > 0,
            img_crop,
            big_image[start_y:end_y, start_x:end_x]
        )
        
        # 如果是target，更新mask
        if is_target:
            full_mask[start_y:end_y, start_x:end_x] = mask_crop
        
        return big_image, full_mask

    def __call__(self, idx=None):
        conditions = self.get_conditions()
        data_dict = self.load_data(idx, conditions["other_class"])
        number_image_mask = self.get_number_mask(conditions["number_list"])
        patch_mask, patch = self._crop_mask_and_patch(data_dict["this_data"]["mask"], data_dict["this_data"]["img"])
        other_mask, other_patch = self._crop_mask_and_patch(data_dict["other_data"]["mask"], data_dict["other_data"]["img"]) \
                                        if data_dict["other_data"]["mask"] is not None else (None, None)
        patch_mask, patch = torch.tensor(patch_mask, dtype=torch.float32).unsqueeze(0), torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)
        other_mask, other_patch = (torch.tensor(other_mask, dtype=torch.float32).unsqueeze(0), torch.tensor(other_patch, dtype=torch.float32).permute(2, 0, 1)) \
                                    if other_mask is not None else (None, None)
        instance_list = [(NumberOcrDataset.put_number_into_image(patch, patch_mask, number_image_mask[i][0], number_image_mask[i][1]), patch_mask)for i in range(len(number_image_mask))]
        text_list = [NumberOcrDataset.generate_txt(data_dict["this_data"]["txt"], conditions["number_list"][i]) for i in range(len(number_image_mask))]
        if other_mask is not None:
            instance_list.append((NumberOcrDataset.put_number_into_image(other_patch, other_mask, number_image_mask[0][0], number_image_mask[0][1]), other_mask))
            text_list.append(NumberOcrDataset.generate_txt(data_dict["other_data"]["txt"], conditions["number_list"][0]))
        big_image, large_mask, _ = self._compose_image(instance_list, target_idx=0)
        
        # full_image = torch.tensor(big_image, dtype=torch.uint8).permute(2, 0, 1)  # (C, H, W)
        # target_mask = torch.tensor(large_mask, dtype=torch.uint8)  # (H, W)
        target_text = text_list[0]  # 第一个数字的文本
        if not self.load_raw_data:
            full_mask_img = Image.fromarray(large_mask.astype(np.uint8)).convert("P")
            img_pil = Image.fromarray(big_image.astype(np.uint8)).convert("RGB")
            img_tensor, mask_tensor = self.apply_transforms(img_pil, full_mask_img)
            input_ids, attention_mask = self.tokenize_text(target_text)
            return img_tensor, mask_tensor, input_ids, attention_mask
        else:
            return big_image, target_text, large_mask

if __name__ == "__main__":
    dataset = NumberOcrDataset(
        prob=1.0,
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc", 
        split="train", 
        max_tokens=20,
        load_raw_data = True,
    )
    
    for i in range(100):
        result = dataset()
        if result is None:
            print(f"[{i+1}] Failed to generate sample.")
            continue

        img, txt, mask = result  # mask 是二值的，只包含目标 instance

        print(f"Referring text: {txt}")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")  # 应该是 [0, 1]

        # Prepare image
        vis_img = img.copy()
        if vis_img.max() <= 1.0:
            vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # Create colored mask for target instance (green)
        target_mask = (mask > 0).astype(np.uint8) * 255
        target_mask_colored = np.zeros_like(vis_img)
        target_mask_colored[:, :, 1] = target_mask  # Green channel

        # Overlay
        overlay = cv2.addWeighted(vis_img, 0.6, target_mask_colored, 0.4, 0)

        # Add text
        cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * vis_img.shape[1] / 640, (255, 255, 255), 2)

        # Save
        save_path = f"visualizations/synthetics/referring_example_{i+1}_number.png"
        cv2.imwrite(save_path, overlay)
        print(f"Saved to {save_path}\n")
    