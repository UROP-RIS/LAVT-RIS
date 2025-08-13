import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager
import math


class NumberGenerator:
    def __init__(self, input_number: int, odd_dict=None):
        if odd_dict is None:
            self.odd_dict = {
                "color_sample": 1.0,
                "horizontal_shear": 0.8,
                "vertical_shear": 0.5,
                "horizontal_distort": 0.5,
                "rotate": 0.5,
                "tilt": 0.8,
                "scale": 0.5,
            }
        else:
            self.odd_dict = {k: max(0, min(1, v)) for k, v in odd_dict.items()}

        self.input_number = input_number
        self.augmentation_list = {
            k: random.random() < v for k, v in self.odd_dict.items()
        }
        self.sequence = [k for k, v in self.augmentation_list.items() if v]
        random.shuffle(self.sequence)

        # Step 1: Generate initial image and mask
        self.image, self.mask = self._generate_initial_image()

        # Step 2: Crop to bounding box
        self.image, self.mask = self._crop_to_bounding_box(self.image, self.mask)

        # Ensure max(H, W) <= 256
        self.image, self.mask = self._resize_if_needed(self.image, self.mask)

    def _get_reliable_font(self):
        """获取支持数字的可靠字体"""
        # 获取系统字体列表
        fonts = font_manager.findSystemFonts()

        # 优先尝试常用的系统字体
        preferred_fonts = [
            "DejaVuSans.ttf",
            "Arial.ttf",
            "Helvetica.ttf",
            "TimesNewRoman.ttf",
            "CourierNew.ttf",
        ]

        for font_name in preferred_fonts:
            for font_path in fonts:
                if font_name.lower() in font_path.lower():
                    try:
                        font_prop = font_manager.FontProperties(fname=font_path)
                        return font_prop
                    except:
                        continue

        # 如果找不到偏好字体，随机选择一个可用字体
        for _ in range(10):  # 最多尝试10次
            try:
                font_path = random.choice(fonts)
                font_prop = font_manager.FontProperties(fname=font_path)
                return font_prop
            except:
                continue

        # 如果都失败了，使用默认字体
        return font_manager.FontProperties()

    def _generate_initial_image(self):
        # 获取可靠的字体
        font_prop = self._get_reliable_font()

        # 生成图像
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.text(
            0.5,
            0.5,
            str(self.input_number),
            fontsize=60,
            ha="center",
            va="center",
            fontproperties=font_prop,
            transform=ax.transAxes,
        )  # 使用相对坐标
        ax.axis("off")

        # 抑制字体警告
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.canvas.draw()

        # Convert to numpy array - 使用正确的API
        buf = np.array(fig.canvas.buffer_rgba())
        img = buf[:, :, :3]  # 去掉alpha通道

        # Create mask (white text on black background)
        mask_fig, mask_ax = plt.subplots(figsize=(4, 4), dpi=100)
        mask_ax.text(
            0.5,
            0.5,
            str(self.input_number),
            fontsize=60,
            color="white",
            ha="center",
            va="center",
            fontproperties=font_prop,
            transform=mask_ax.transAxes,
        )
        mask_ax.axis("off")
        mask_fig.patch.set_facecolor("black")

        # 抑制字体警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_fig.canvas.draw()

        mask_buf = np.array(mask_fig.canvas.buffer_rgba())
        mask = (mask_buf[:, :, :3].mean(axis=2) > 0).astype(np.uint8)

        plt.close("all")

        # Convert to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.tensor(img).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()  # Add channel dim

        return image_tensor, mask_tensor

    def _crop_to_bounding_box(self, image, mask):
        mask_np = mask.squeeze().numpy()
        coords = np.column_stack(np.where(mask_np > 0))
        if coords.size == 0:
            return image, mask
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        image_cropped = image[:, y_min : y_max + 1, x_min : x_max + 1]
        mask_cropped = mask[:, y_min : y_max + 1, x_min : x_max + 1]

        return image_cropped, mask_cropped

    def _resize_if_needed(self, image, mask):
        _, H, W = image.shape
        max_dim = max(H, W)
        if max_dim != 256:
            scale = 256 / max_dim
            new_H, new_W = int(H * scale), int(W * scale)
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(new_H, new_W), mode="bilinear"
            )[0]
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), size=(new_H, new_W), mode="nearest"
            )[0]
        return image, mask

    @staticmethod
    def color_sample(image, mask):
        color = torch.randint(0, 256, (3, 1, 1)).float()
        image = image * 0 + color
        return image, mask

    @staticmethod
    def horizontal_shear(image, mask):
        C, H, W = image.shape
        shear = random.uniform(-W / 2, W / 2)
        M = np.array([[1, shear / H, 0], [0, 1, 0]], dtype=np.float32)
        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = mask.squeeze().numpy()

        # 计算边界框以确保所有内容都被包含
        corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]]).T
        transformed_corners = M @ corners
        min_x, min_y = transformed_corners[0].min(), transformed_corners[1].min()
        max_x, max_y = transformed_corners[0].max(), transformed_corners[1].max()

        new_W = int(max_x - min_x)
        new_H = int(max_y - min_y)

        # 调整变换矩阵以适应新的画布
        M[0, 2] = -min_x
        M[1, 2] = -min_y

        image_sheared = cv2.warpAffine(
            image_np, M, (new_W, new_H), flags=cv2.INTER_LINEAR
        )
        mask_sheared = cv2.warpAffine(
            mask_np, M, (new_W, new_H), flags=cv2.INTER_NEAREST
        )

        image_t = torch.tensor(image_sheared).permute(2, 0, 1).float()
        mask_t = torch.tensor(mask_sheared).unsqueeze(0).float()
        return image_t, mask_t

    @staticmethod
    def vertical_shear(image, mask):
        C, H, W = image.shape
        shear = random.uniform(-H / 2, H / 2)
        M = np.array([[1, 0, 0], [shear / W, 1, 0]], dtype=np.float32)
        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = mask.squeeze().numpy()

        # 计算边界框以确保所有内容都被包含
        corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]]).T
        transformed_corners = M @ corners
        min_x, min_y = transformed_corners[0].min(), transformed_corners[1].min()
        max_x, max_y = transformed_corners[0].max(), transformed_corners[1].max()

        new_W = int(max_x - min_x)
        new_H = int(max_y - min_y)

        # 调整变换矩阵以适应新的画布
        M[0, 2] = -min_x
        M[1, 2] = -min_y

        image_sheared = cv2.warpAffine(
            image_np, M, (new_W, new_H), flags=cv2.INTER_LINEAR
        )
        mask_sheared = cv2.warpAffine(
            mask_np, M, (new_W, new_H), flags=cv2.INTER_NEAREST
        )

        image_t = torch.tensor(image_sheared).permute(2, 0, 1).float()
        mask_t = torch.tensor(mask_sheared).unsqueeze(0).float()
        return image_t, mask_t

    @staticmethod
    def horizontal_distort(image, mask):
        C, H, W = image.shape
        period = random.uniform(H, 4 * H)
        amplitude = random.uniform(0, min(H, W) / 8)
        phase = random.uniform(0, 2 * math.pi)

        dx = amplitude * np.sin(2 * math.pi * np.arange(H) / period + phase)
        map_x = np.tile(np.arange(W), (H, 1)) + dx[:, None]
        map_y = np.tile(np.arange(H)[:, None], (1, W))

        # 计算新的边界
        min_x = map_x.min()
        max_x = map_x.max()

        # 调整输出图像大小
        left_padding = max(0, int(np.ceil(-min_x)))
        right_padding = max(0, int(np.ceil(max_x - (W - 1))))

        new_W = W + left_padding + right_padding

        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = mask.squeeze().numpy()

        # 扩展原图
        if left_padding > 0 or right_padding > 0:
            extended_image = np.pad(
                image_np,
                ((0, 0), (left_padding, right_padding), (0, 0)),
                mode="constant",
            )
            extended_mask = np.pad(
                mask_np, ((0, 0), (left_padding, right_padding)), mode="constant"
            )
            image_np = extended_image
            mask_np = extended_mask
            map_x = map_x + left_padding

        remapped_image = cv2.remap(
            image_np,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
        )
        remapped_mask = cv2.remap(
            mask_np,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
        )

        image_t = torch.tensor(remapped_image).permute(2, 0, 1).float()
        mask_t = torch.tensor(remapped_mask).unsqueeze(0).float()
        return image_t, mask_t

    @staticmethod
    def rotate(image, mask):
        angle = random.uniform(-10, 10)
        C, H, W = image.shape
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_W = int((H * sin) + (W * cos))
        new_H = int((H * cos) + (W * sin))
        M[0, 2] += (new_W / 2) - center[0]
        M[1, 2] += (new_H / 2) - center[1]

        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = mask.squeeze().numpy()

        rotated_image = cv2.warpAffine(
            image_np, M, (new_W, new_H), flags=cv2.INTER_LINEAR
        )
        rotated_mask = cv2.warpAffine(
            mask_np, M, (new_W, new_H), flags=cv2.INTER_NEAREST
        )

        image_t = torch.tensor(rotated_image).permute(2, 0, 1).float()
        mask_t = torch.tensor(rotated_mask).unsqueeze(0).float()
        return image_t, mask_t

    @staticmethod
    def tilt(image, mask):
        C, H, W = image.shape
        pts1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        offset = random.uniform(-W / 4, W / 4)
        pts2 = np.float32([[0, 0], [W, 0], [offset, H], [W + offset, H]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = mask.squeeze().numpy()

        # 计算输出尺寸以包含所有内容
        corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M)
        x_coords = transformed_corners[:, :, 0].flatten()
        y_coords = transformed_corners[:, :, 1].flatten()

        min_x, max_x = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
        min_y, max_y = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))

        new_W = max_x - min_x
        new_H = max_y - min_y

        # 调整变换矩阵
        M_trans = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
        )
        M = M_trans @ M

        tilted_image = cv2.warpPerspective(
            image_np, M, (new_W, new_H), flags=cv2.INTER_LINEAR
        )
        tilted_mask = cv2.warpPerspective(
            mask_np, M, (new_W, new_H), flags=cv2.INTER_NEAREST
        )

        image_t = torch.tensor(tilted_image).permute(2, 0, 1).float()
        mask_t = torch.tensor(tilted_mask).unsqueeze(0).float()
        return image_t, mask_t

    @staticmethod
    def scale(image, mask):
        factor = random.uniform(0.3, 1.5)
        C, H, W = image.shape
        new_H, new_W = int(H * factor), int(W * factor)
        image_scaled = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(new_H, new_W), mode="bilinear"
        )[0]
        mask_scaled = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(new_H, new_W), mode="nearest"
        )[0]
        return image_scaled, mask_scaled

    def aug(self):
        for aug_name in self.sequence:
            func = getattr(self, aug_name)
            self.image, self.mask = func(self.image, self.mask)
        self.image, self.mask = self._crop_to_bounding_box(self.image, self.mask)
        self.image, self.mask = self._resize_if_needed(self.image, self.mask)
        self.image[self.mask.repeat(3, 1, 1) == 0] = 255

    def save_visualization(self, save_path):
        """保存图像和mask的可视化结果"""
        import matplotlib.pyplot as plt

        # 转换tensor为numpy数组
        image_np = self.image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = self.mask.squeeze().numpy()

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # 显示原图像
        ax1.imshow(image_np)
        ax1.set_title("Image")
        ax1.axis("off")

        # 显示mask (使用灰度colormap)
        ax2.imshow(mask_np, cmap="gray")
        ax2.set_title("Mask")
        ax2.axis("off")

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()


if __name__ == "__main__":
    num_gen = NumberGenerator(13)
    num_gen.aug()
    print(
        num_gen.image.shape, num_gen.mask.shape
    )  # torch.Size([3, 31, 20]) torch.Size([1, 31, 20])

    num_gen.save_visualization("/data/datasets/tzhangbu/LAVT-RIS/visualizations/visualization.png")
    print("Visualization saved to visualization.png")