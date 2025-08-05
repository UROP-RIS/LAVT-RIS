import datetime
import os
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


def evaluate_pseudo_candidate(model, data_loader, bert_model, device, dataset):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    eval_seg_iou_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    
    cumI, cumU = 0, 0

    header = 'Test:'

    with torch.no_grad():
        for idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            images, targets, sentences, attentions = data['img'], data['target'], data['txt'], data['attention_mask']
            images = images.to(device)
            sentences = sentences.to(device)
            attentions = attentions.to(device)
            B = images.size(0)

            raw_items = [dataset.get_raw_item(i + idx * B) for i in range(B)]

            for b in range(B):
                sentence = sentences[b]          # (1, L) 
                attention = attentions[b]        # (1, L)

                if bert_model is not None:
                    last_hidden_states = bert_model(sentence, attention_mask=attention)[0]
                    embedding = last_hidden_states.permute(0, 2, 1)  # (1, D, L)
                    l_mask = attention.unsqueeze(-1)  # (1, L, 1) ✅ 正确形状
                    output = model(images[b:b+1], embedding, l_mask=l_mask)
                else:
                    output = model(images[b:b+1], sentence, l_mask=attention.unsqueeze(-1))

                # Get predicted mask: (1, 2, H_model, W_model) -> (H_model, W_model)
                pred_mask = F.softmax(output, dim=1).argmax(1).cpu().squeeze(0).numpy()  # (H_model, W_model)

                # Get original size
                raw_item = raw_items[b]
                orig_h, orig_w = raw_item["orig_size"]

                # Resize pred_mask to original resolution
                pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
                pred_mask_orig = pred_mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
                pred_mask_np = np.array(pred_mask_orig).astype(bool)  # (orig_h, orig_w)

                all_masks = raw_item["all_masks"]  # already (orig_h, orig_w)
                gt_mask = raw_item["gt"]  # (orig_h, orig_w)

                # Step 1: find best candidate that matches resized pred_mask
                best_iou = -1
                best_candidate = None
                for cand in all_masks:
                    i, u = computeIoU(pred_mask_np, cand)
                    iou = i / u if u > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate = cand

                if best_candidate is None or best_iou < 0.4:
                    # best_candidate = np.zeros_like(pred_mask_np)
                    best_candidate = pred_mask_np  # Fallback to pred_mask if no candidate found


                # Step 2: compare best candidate with GT
                I, U = computeIoU(best_candidate, gt_mask)
                cumI += I
                cumU += U
                this_iou = I * 1.0 / U if U > 0 else 0.0

                mean_IoU.append(this_iou)
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou_list[n_eval_iou])
                seg_total += 1

    # Final results
    mIoU = np.mean(mean_IoU) if mean_IoU else 0.0
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (float(cumI) / float(cumU) * 100.)
    print(results_str)

    return mIoU


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
    evaluate_pseudo_candidate(model, data_loader_test, bert_model, device, dataset_test)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
