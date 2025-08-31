import numpy as np
import torch.nn.functional as F
import cv2


def get_connected_components(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    num_labels, labeled_img = cv2.connectedComponents(mask)
    unique, counts = np.unique(labeled_img, return_counts=True)
    area_map = np.zeros(num_labels)
    for lbl, cnt in zip(unique, counts):
        area_map[lbl] = cnt
    areas = area_map[labeled_img]
    return labeled_img, areas


def postprocess_binary_mask(
    mask: np.ndarray,
    max_hole_area: int = 0,
    max_sprinkle_area: int = 0,
) -> np.ndarray:
    if mask.dtype != bool:
        mask = (mask > 0).astype(bool)

    cleaned = mask.copy()

    if max_hole_area > 0:
        background = ~cleaned
        labels, areas = get_connected_components(background)
        is_small_hole = (labels > 0) & (areas <= max_hole_area)
        cleaned[is_small_hole] = True

    if max_sprinkle_area > 0:
        labels, areas = get_connected_components(cleaned)
        is_small_sprinkle = (labels > 0) & (areas <= max_sprinkle_area)
        cleaned[is_small_sprinkle] = False

    return cleaned

def fill_holes_in_components(mask: np.ndarray) -> np.ndarray:
    original_dtype = mask.dtype
    mask = (mask > 0).astype(np.uint8)
    cleaned = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cleaned, contours, -1, 1, thickness=cv2.FILLED)
    return cleaned.astype(original_dtype)

def count_connected_components(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)
    return num_labels - 1  # 减去背景 (label 0)


def create_test_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    mask[30:40, 30:40] = False
    mask[50:53, 50:53] = False
    mask[10:13, 10:13] = True
    mask[80:82, 80:82] = True
    return mask


if __name__ == "__main__":
    test_mask = create_test_mask()

    print("✅ 原始 mask 统计:")
    print(f"  总前景像素: {test_mask.sum()}")

    clean_mask = postprocess_binary_mask(
        test_mask,
        max_hole_area=10,
        max_sprinkle_area=10
    )

    print("\n✅ 清理后 mask 统计:")
    print(f"  总前景像素: {clean_mask.sum()}")
    
    # 使用
    print(f"  前景连通域数量: {count_connected_components(test_mask)}")
    print(f"  前景连通域数量: {count_connected_components(clean_mask)}")
    
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    mask[30:40, 30:40] = False  # 孔洞
    mask[50:53, 50:53] = False  # 小孔

    filled = fill_holes_in_components(mask)
    print(filled[35, 35])  # True ✅ 孔被填了