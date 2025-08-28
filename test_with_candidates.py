
import torch
import torch.utils.data
from bert.modeling_bert import BertModel
from lib import segmentation
import transforms as T
import utils
import numpy as np
import torch.nn.functional as F
import cv2
import os
from matplotlib import cm
from data.dataset_pseudo import PseudoLabelDataset
from misc.common import make_object_from_config
import json

import tqdm

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))
    return I, U


def evaluate_pseudo_candidate(model, data_loader, bert_model, device, output_dir, stream_configs):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # IoU 阈值
    eval_seg_iou_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    # 初始化两套指标
    seg_correct_pseudo = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    mean_IoU_pseudo = []
    cumI_pseudo, cumU_pseudo = 0, 0

    seg_correct_cand = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    mean_IoU_cand = []
    cumI_cand, cumU_cand = 0, 0

    seg_total = 0  # 样本总数

    header = 'Test:'

    vis_freq = stream_configs.get("vis_freq", 50)
    enable_vis = stream_configs.get("enable_visualization", False)  # 是否启用可视化
    save_new_label = stream_configs.get("save_new_label", False)  # 是否保存新的伪标签
    if enable_vis:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Visualization saved to: {output_dir}")

    with torch.no_grad():
        for idx, data in tqdm.tqdm(enumerate(metric_logger.log_every(data_loader, 100, header))):
            images = data['img'].to(device)                 # (B, 3, H_t, W_t)
            sentences = data['txt'].to(device).squeeze(1)              # (B, 1, max_tokens)
            attentions = data['attention_mask'].to(device).squeeze(1)  # (B, 1, max_tokens)
            B = images.size(0)
            # 获取原始数据（list of numpy arrays）
            raw_imgs = data['raw_img']        # list[B], each (H_i, W_i, 3)
            gts = data['gt']                  # list[B], each (H_i, W_i) or (N, H_i, W_i)
            all_masks_list = data['all_masks']  # list[B], each is list of (H_i, W_i)
            txts_raw = data['txt_raw']        # list[B] of str
            orig_sizes = data['orig_size']    # list[B] of (H_i, W_i)
            # ----------------------------
            # 🔹 模型推理（Batched）
            # ----------------------------
            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                l_mask = attentions.unsqueeze(-1)
                output = model(images, embedding, l_mask=l_mask)
            else:
                output = model(images, sentences.squeeze(1), l_mask=attentions.squeeze(1))

            # 获取预测分数
            pred_scores = F.softmax(output, dim=1)[:, 1].cpu().numpy()  # (B, H_t, W_t)

            # ----------------------------
            # 🔹 逐样本后处理（Resize + IoU 计算）
            # ----------------------------
            for b in range(B):
                pred_score = pred_scores[b]  # (H_t, W_t)
                raw_img = raw_imgs[b]
                orig_h, orig_w = orig_sizes[b]
                gt_mask = gts[b]
                all_masks = all_masks_list[b]
                sentence_str = txts_raw[b]

                # Resize 到原始分辨率
                pred_mask = cv2.resize((pred_score > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                pred_score_vis = cv2.resize(pred_score, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                # gt_mask = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                # all_masks = [cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) for m in all_masks]

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

                # === Step 2: Pseudo Label vs GT ===
                I_p, U_p = computeIoU(pred_mask, gt_mask)
                iou_p = I_p / U_p if U_p > 0 else 0.0
                cumI_pseudo += I_p
                cumU_pseudo += U_p
                mean_IoU_pseudo.append(iou_p)
                for i, thres in enumerate(eval_seg_iou_list):
                    seg_correct_pseudo[i] += (iou_p >= thres)

                # === Step 3: Best Candidate vs GT ===
                I_c, U_c = computeIoU(best_candidate, gt_mask)
                iou_c = I_c / U_c if U_c > 0 else 0.0
                cumI_cand += I_c
                cumU_cand += U_c
                mean_IoU_cand.append(iou_c)
                for i, thres in enumerate(eval_seg_iou_list):
                    seg_correct_cand[i] += (iou_c >= thres)

                seg_total += 1

                # ================================
                # ✅ 可视化（可选抽样）
                # ================================
                iter_ = idx * B + b
                if iter_ % vis_freq == 0:
                    img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

                    conf_norm = (pred_score_vis - pred_score_vis.min()) / (pred_score_vis.max() - pred_score_vis.min() + 1e-8)
                    heatmap_rgb = cm.jet(conf_norm)[:, :, :3]
                    heatmap_bgr = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # --- Mask 叠加 ---
                    def draw_mask(img, mask, color):
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
                        return cv2.resize(x, new_size, interpolation=cv2.INTER_LINEAR)

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
    # ✅ 最终评估结果输出
    # ================================
    mIoU_pseudo = np.mean(mean_IoU_pseudo) if mean_IoU_pseudo else 0.0
    overall_IoU_pseudo = cumI_pseudo / cumU_pseudo if cumU_pseudo > 0 else 0.0

    mIoU_cand = np.mean(mean_IoU_cand) if mean_IoU_cand else 0.0
    overall_IoU_cand = cumI_cand / cumU_cand if cumU_cand > 0 else 0.0

    print('\n' + '='*70)
    print('✅ FINAL EVALUATION RESULTS')
    print('='*70)

    print('📌 Pseudo Label vs Ground Truth')
    print(f'  mIoU: {mIoU_pseudo * 100:.2f}%')
    print(f'  Overall IoU: {overall_IoU_pseudo * 100:.2f}%')
    for thres, val in zip(eval_seg_iou_list, seg_correct_pseudo):
        print(f'  Precision@{thres:.1f}: {val / seg_total * 100:.2f}%')

    print()

    print('📌 Best Candidate vs Ground Truth')
    print(f'  mIoU: {mIoU_cand * 100:.2f}%')
    print(f'  Overall IoU: {overall_IoU_cand * 100:.2f}%')
    for thres, val in zip(eval_seg_iou_list, seg_correct_cand):
        print(f'  Precision@{thres:.1f}: {val / seg_total * 100:.2f}%')

    print('='*70)

    return mIoU_pseudo


def main(args):
    device = torch.device(args.device)
    configs = json.load(open(args.configs, 'r'))
    dataset = make_object_from_config(configs["dataset"])
    test_sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=36, 
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=PseudoLabelDataset.eval_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    # 模型加载（保持不变）
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)
    
    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            bert_model.pooler = None
        bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = bert_model.to(device)
    else:
        bert_model = None
    

    resume_ckpt_dir = os.path.dirname(args.resume)
    output_dir = f"{resume_ckpt_dir}/{args.dataset}_{args.split}_{args.model}"
    evaluate_pseudo_candidate(model, data_loader_test, bert_model, device, output_dir=output_dir, stream_configs=configs.get("stream_configs", {}))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
