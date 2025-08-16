def compute_iou_xyxy(bbox1: list, bbox2: list):
    """
    Compute Intersection over Union (IoU) for two bounding boxes in xyxy format.
    
    Args:
        bbox1 (list): Bounding box 1 in the format [x1, y1, x2, y2].
        bbox2 (list): Bounding box 2 in the format [x1, y1, x2, y2].
    
    Returns:
        float: IoU value between bbox1 and bbox2.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area_bbox1 + area_bbox2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def compute_iou_xywh(bbox1: list, bbox2: list):
    x1 = bbox1[0] - bbox1[2] / 2
    y1 = bbox1[1] - bbox1[3] / 2
    x2 = bbox1[0] + bbox1[2] / 2  # 修正
    y2 = bbox1[1] + bbox1[3] / 2  # 修正

    x3 = bbox2[0] - bbox2[2] / 2
    y3 = bbox2[1] - bbox2[3] / 2
    x4 = bbox2[0] + bbox2[2] / 2  # 修正
    y4 = bbox2[1] + bbox2[3] / 2  # 修正

    return compute_iou_xyxy([x1, y1, x2, y2], [x3, y3, x4, y4])