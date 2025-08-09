import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt
import os

from matplotlib import cm



def get_dataset(image_set, transform, args):
    # 使用你的 PseudoLabelDataset
    from data.dataset_pseudo import PseudoLabelDataset
    
    ds = PseudoLabelDataset(
        image_transforms=transform,
        root=args.data_root if hasattr(args, 'data_root') else "/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset=args.dataset if hasattr(args, 'dataset') else "unc",
        split=image_set,
        max_tokens=args.max_tokens if hasattr(args, 'max_tokens') else 20,
        augment_text_root=args.augment_text_root if hasattr(args, 'augment_text_root') else f"augmentation/data/{args.dataset}/{image_set}",
        eval_mode=True
    )
    num_classes = 2
    return ds, num_classes


def evaluate_pseudo_candidate(model, data_loader, bert_model, device, dataset, output_dir="visualizations"):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # IoU 阈值
    eval_seg_iou_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    # 初始化两套指标
    # --- Pseudo Label vs GT ---
    seg_correct_pseudo = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    mean_IoU_pseudo = []
    cumI_pseudo, cumU_pseudo = 0, 0

    # --- Best Candidate vs GT ---
    seg_correct_cand = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    mean_IoU_cand = []
    cumI_cand, cumU_cand = 0, 0

    seg_total = 0  # 样本总数

    header = 'Test:'

    os.makedirs(output_dir, exist_ok=True)
    print(f"📝 Saving visualizations to: {output_dir}")

    VIS_FREQ = 50  # 可调：每 N 个 batch 保存一次可视化

    with torch.no_grad():
        for idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            images, sentences, attentions = data['img'], data['txt'], data['attention_mask']
            images = images.to(device)
            sentences = sentences.to(device)
            attentions = attentions.to(device)
            B = images.size(0)

            raw_items = [dataset.get_raw_item(i + idx * B) for i in range(B)]

            for b in range(B):
                sentence = sentences[b]
                attention = attentions[b]

                if bert_model is not None:
                    last_hidden_states = bert_model(sentence, attention_mask=attention)[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    l_mask = attention.unsqueeze(-1)
                    output = model(images[b:b+1], embedding, l_mask=l_mask)
                else:
                    output = model(images[b:b+1], sentence, l_mask=attention.unsqueeze(-1))

                # 获取模型输出
                pred_score = F.softmax(output, dim=1)[0, 1].cpu().squeeze(0).numpy()
                pred_mask = (pred_score > 0.5).astype(bool)  # Pseudo label

                raw_item = raw_items[b]
                raw_img = raw_item["raw_img"]
                orig_h, orig_w = raw_img.shape[:2]
                sentence_str = raw_item["txt"]

                # Resize 函数
                def resize_mask(mask, size):
                    return np.array(Image.fromarray(mask.astype(np.uint8)).resize(size, resample=Image.NEAREST))

                def resize_heatmap(heatmap, size):
                    return np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(size, resample=Image.BILINEAR))

                # Resize 到原始图像分辨率
                pred_mask = resize_mask(pred_mask, (orig_w, orig_h))
                pred_score = resize_heatmap(pred_score, (orig_w, orig_h))
                gt_mask = resize_mask(raw_item["gt"], (orig_w, orig_h))

                # 所有候选 mask
                all_masks = [resize_mask(m, (orig_w, orig_h)) for m in raw_item["all_masks"]]

                # === Step 1: 找最佳候选 ===
                best_iou = -1
                best_candidate = None
                for cand in all_masks:
                    i, u = computeIoU(pred_mask, cand)
                    iou = i / u if u > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate = cand
                if best_candidate is None:
                    best_candidate = pred_mask

                # === Step 2: 计算 Pseudo Label vs GT ===
                I_p, U_p = computeIoU(pred_mask, gt_mask)
                iou_p = I_p / U_p if U_p > 0 else 0.0
                cumI_pseudo += I_p
                cumU_pseudo += U_p
                mean_IoU_pseudo.append(iou_p)
                for i, thres in enumerate(eval_seg_iou_list):
                    seg_correct_pseudo[i] += (iou_p >= thres)

                # === Step 3: 计算 Best Candidate vs GT ===
                I_c, U_c = computeIoU(best_candidate, gt_mask)
                iou_c = I_c / U_c if U_c > 0 else 0.0
                cumI_cand += I_c
                cumU_cand += U_c
                mean_IoU_cand.append(iou_c)
                for i, thres in enumerate(eval_seg_iou_list):
                    seg_correct_cand[i] += (iou_c >= thres)

                seg_total += 1  # 样本计数 +1

                # ================================
                # ✅ 可视化（可选抽样）
                # ================================
                if idx % VIS_FREQ == 0 and b == 0:
                    img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

                    # --- 热力图 ---
                    conf_norm = (pred_score - pred_score.min()) / (pred_score.max() - pred_score.min() + 1e-8)
                    heatmap_rgb = cm.jet(conf_norm)[:, :, :3]
                    heatmap_bgr = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                    # --- Mask 叠加 ---
                    def draw_mask(img, mask, color, alpha=0.5):
                        # print(f"[Debug] pred_mask sum: {mask.sum()} / {mask.size} ({mask.sum() / mask.size * 100:.2f}%)")
                        mask = mask.astype(bool)
                        overlay = img.copy()
                        overlay[mask] = color
                        return overlay

                    img_pseudo = draw_mask(img_bgr.copy(), pred_mask, (0, 0, 255))     # Red
                    img_cand = draw_mask(img_bgr.copy(), best_candidate, (0, 255, 255)) # Yellow
                    img_gt = draw_mask(img_bgr.copy(), gt_mask, (0, 255, 0))           # Green

                    # --- 分辨率适配 ---
                    H, W = img_bgr.shape[:2]
                    scale = 600 / H if H > 600 else 1
                    new_size = (int(W * scale), int(H * scale))

                    def resize(x):
                        return cv2.resize(x, new_size, interpolation=cv2.INTER_CUBIC)

                    img_bgr_r = resize(img_bgr)
                    heatmap_bgr_r = resize(heatmap_bgr)
                    img_pseudo_r = resize(img_pseudo)
                    img_cand_r = resize(img_cand)
                    img_gt_r = resize(img_gt)

                    # --- 拼接 ---
                    row1 = np.hstack([img_bgr_r, heatmap_bgr_r])
                    row2 = np.hstack([img_pseudo_r, img_cand_r])
                    row3 = np.hstack([img_gt_r, np.zeros_like(img_gt_r)])
                    combined = np.vstack([row1, row2, row3])

                    # --- 文本 ---
                    text_area = np.ones((80, combined.shape[1], 3), dtype=np.uint8) * 255
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale, thickness = 0.6, 1

                    line1 = f"Text: {sentence_str}"
                    line2 = f"Pseudo IoU: {iou_p:.3f} | Cand IoU: {iou_c:.3f}"

                    cv2.putText(text_area, line1, (10, 25), font, font_scale, (0, 0, 0), thickness)
                    cv2.putText(text_area, line2, (10, 55), font, font_scale, (0, 0, 0), thickness)

                    vis_img = np.vstack([combined, text_area])
                    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)

                    vis_path = os.path.join(output_dir, f"vis_batch{idx:04d}_img{b}.jpg")
                    cv2.imwrite(vis_path, vis_img)
                    print(f"🎨 Saved visualization: {vis_path}")

    # ================================
    # ✅ 最终评估结果输出（双指标）
    # ================================
    mIoU_pseudo = np.mean(mean_IoU_pseudo) if mean_IoU_pseudo else 0.0
    overall_IoU_pseudo = cumI_pseudo / cumU_pseudo if cumU_pseudo > 0 else 0.0

    mIoU_cand = np.mean(mean_IoU_cand) if mean_IoU_cand else 0.0
    overall_IoU_cand = cumI_cand / cumU_cand if cumU_cand > 0 else 0.0

    print('\n' + '='*70)
    print('✅ FINAL EVALUATION RESULTS')
    print('='*70)

    # --- Pseudo Label vs GT ---
    print('📌 Pseudo Label vs Ground Truth')
    print(f'  mIoU: {mIoU_pseudo * 100:.2f}%')
    print(f'  Overall IoU: {overall_IoU_pseudo * 100:.2f}%')
    for thres, val in zip(eval_seg_iou_list, seg_correct_pseudo):
        print(f'  Precision@{thres:.1f}: {val / seg_total * 100:.2f}%')

    print()

    # --- Best Candidate vs GT ---
    print('📌 Best Candidate vs Ground Truth')
    print(f'  mIoU: {mIoU_cand * 100:.2f}%')
    print(f'  Overall IoU: {overall_IoU_cand * 100:.2f}%')
    for thres, val in zip(eval_seg_iou_list, seg_correct_cand):
        print(f'  Precision@{thres:.1f}: {val / seg_total * 100:.2f}%')

    print('='*70)

    # 返回 pseudo mIoU（可用于主函数记录）
    return mIoU_pseudo
    return mIoU_pseudo


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    
    # Use your custom dataset
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1,  # 推荐 batch_size=1，便于与 get_raw_item 对齐
        sampler=test_sampler, 
        num_workers=args.workers
    )

    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    # 使用新 evaluation 函数
    resume_ckpt_name = os.path.basename(args.resume)
    evaluate_pseudo_candidate(model, data_loader_test, bert_model, device, dataset_test, output_dir=f"./visualizations/{resume_ckpt_name}_{args.dataset}_{args.split}_{args.model}")


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
