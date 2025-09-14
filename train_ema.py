import datetime
from torch.cuda.amp import autocast, GradScaler
from utils import NativeScalerWithGradNormCount
import os
import time
import torch
import torch.utils.data
from data.transforms import cross_align_features
from functools import reduce
import operator
from bert.modeling_bert import BertModel
from lib import segmentation
import transforms as T
import utils
import numpy as np
import json
from utils import PartialDistributedSampler
from misc.common import make_object_from_config
from misc.workspace import create_workspace, save_configs_and_args
from torch.utils.tensorboard import SummaryWriter
from misc.ema import update_teacher_model



# ----------------------- 重要修改开始 -----------------------
# 我们不再通过 argparse 从命令行获取 local_rank
# 而是在 main() 函数中从环境变量读取
def get_args_parser():
    parser = get_parser()  # 假设 get_parser() 来自你的 args.py
    # 注意：这里不再添加 --local_rank 参数
    return parser
# ----------------------- 重要修改结束 -----------------------

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2
    return ds, num_classes

# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)

def evaluate(model, data_loader, bert_model, writer=None, epoch=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0
    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            with torch.no_grad():
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                    embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                    attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                    output = model(image, embedding, l_mask=attentions)["out"]
                else:
                    output = model(image, sentences, l_mask=attentions)["out"]
            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        precision = seg_correct[n_eval_iou] * 100. / seg_total
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), precision)
        if writer is not None and epoch is not None:
            writer.add_scalar(f"val/precision@{eval_seg_iou_list[n_eval_iou]}", precision, epoch)
            
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)
    
    mIoU, oIoU = 100 * mIoU, 100 * cum_I / cum_U
    
    if writer is not None and epoch is not None:
        writer.add_scalar("val/mean_IoU", mIoU, epoch)
        writer.add_scalar("val/overall_IoU", oIoU, epoch)
    return mIoU, oIoU
    
def freeze_model(model, bert):
    for param in model.parameters():
        param.requires_grad = False
    if bert is not None:
        for param in bert.parameters():
            param.requires_grad = False

def train_one_epoch(model_t,
                    model_s, 
                    bert_t,
                    bert_s,
                    l1,   ## Supervised label loss
                    l2,   ## Unsupervised teacher's label loss
                    l3,   ## Unsupervised token consistent loss
                    l_weights,
                    keep_rate,
                    optimizer, 
                    loss_scaler,
                    data_loader, 
                    lr_scheduler, 
                    epoch, 
                    print_freq,
                    iterations, 
                    writer=None, 
                    stream_configs=None):
    model_t.eval()
    model_s.train()
    bert_t.eval()
    bert_s.train()
    freeze_model(model_t, bert_t)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('label_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('target_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('distilled_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        
        data_t = data['teacher']
        img_t = data_t['image']
        label_t = data_t['mask']
        input_ids_t = data_t['input_ids']
        attentions_t = data_t['attention_mask']
        inv_t = data_t['inv']
        
        data_s = data['student']
        img_s = data_s['image']
        label_s = data_s['mask']
        input_ids_s = data_s['input_ids']
        attentions_s = data_s['attention_mask']
        inv_s = data_s['inv']
        
        sup_loss_weight = data["sup_loss_weight"] # B,
        
        # move to gpu
        img_t = img_t.cuda(non_blocking=True)
        label_t = label_t.cuda(non_blocking=True)
        input_ids_t = input_ids_t.cuda(non_blocking=True).squeeze(1)
        attentions_t = attentions_t.cuda(non_blocking=True).squeeze(1)
        inv_t = inv_t.cuda(non_blocking=True)
        
        img_s = img_s.cuda(non_blocking=True)
        label_s = label_s.cuda(non_blocking=True)
        input_ids_s = input_ids_s.cuda(non_blocking=True).squeeze(1)
        attentions_s = attentions_s.cuda(non_blocking=True).squeeze(1)
        inv_s = inv_s.cuda(non_blocking=True)
        
        sup_loss_weight = sup_loss_weight.cuda(non_blocking=True)
    
        # Teacher inference
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                last_hidden_states_t = bert_t(input_ids_t, attention_mask=attentions_t)[0]
                embedding_t = last_hidden_states_t.permute(0, 2, 1)
                l_mask_t = attentions_t.unsqueeze(-1)
                out_t = model_t(img_t, embedding_t, l_mask=l_mask_t)
        
        # Student
            last_hidden_states_s = bert_s(input_ids_s, attention_mask=attentions_s)[0]
            embedding_s = last_hidden_states_s.permute(0, 2, 1)
            l_mask_s = attentions_s.unsqueeze(-1)
            out_s = model_s(img_s, embedding_s, l_mask=l_mask_s)
            
        ## Supervised loss for labeled data
        if stream_configs["label_supervision"] == "filtered":
            threshold = stream_configs["label_supervision_threshold"]
            valid_label_mask = (sup_loss_weight > threshold).float() # B,
            sup_loss_weight = sup_loss_weight * valid_label_mask

        av_pixel_loss = l1(out_s['out'], label_s, reduce=None).mean(dim=(1,2))  # (B, H, W) -> (B,)
        label_loss = (av_pixel_loss * sup_loss_weight).mean(dim=0)
        
        ## Teacher label loss
        aligned_l_t = cross_align_features(
            teacher_feat = out_t['out'].detach(), teacher_inv = inv_t, student_feat= out_s['out'], student_inv = inv_s, mode='bilinear', align_corners=True 
        )
        target_loss = l2(aligned_l_t, out_s['out'])
        
        ## Distilled loss for tokens
        ## Only regularized the last layer first
        x4_t = out_t['x_c4']
        x4_s = out_s['x_c4']
        
        ## Apply another inverse matrix: inverse scalar of the pixel to the pixel map
        scale_t = x4_t.shape[-1] / 480.0
        scale_s = x4_s.shape[-1] / 480.0
        scale_t_m = torch.tensor([[1.0 / scale_t, 0, 0], 
                                  [0, 1.0 / scale_t, 0], 
                                  [0, 0, 1]]).cuda().unsqueeze(0).repeat(x4_t.shape[0], 1, 1)
        scale_s_m = torch.tensor([[1.0 / scale_s, 0, 0], 
                                  [0, 1.0 / scale_s, 0], 
                                  [0, 0, 1]]).cuda().unsqueeze(0).repeat(x4_s.shape[0], 1, 1)
        
        inv_t_x4 = torch.bmm(inv_t, scale_t_m)
        inv_s_x4 = torch.bmm(inv_s, scale_s_m)
        
        aligned_x4_t = cross_align_features(
            teacher_feat = x4_t, teacher_inv = inv_t_x4, student_feat= x4_s, student_inv = inv_s_x4, mode='bilinear', align_corners=True 
        )
        # aligned_x4_t = x4_t  # 取消对 token 特征的对齐
        distilled_loss = l3(aligned_x4_t.detach(), x4_s)
        total_loss = label_loss * l_weights[0] + target_loss * l_weights[1] + distilled_loss * l_weights[2]
        
        ## Back propagation
        # all_params = list(model_s.parameters()) + list(bert_s.parameters())
        loss_scaler(total_loss, optimizer=optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True)
        lr_scheduler.step()
        
        ## EMA update
        update_teacher_model(student=model_s, teacher=model_t, keep_rate=keep_rate)
        update_teacher_model(student=bert_s, teacher=bert_t, keep_rate=keep_rate)
        metric_logger.update(
            loss=total_loss.item(),
            label_loss=label_loss.item(),
            target_loss=target_loss.item(),
            distilled_loss=distilled_loss.item(),
            lr=optimizer.param_groups[0]["lr"]
        )
        
        if writer is not None:
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            writer.add_scalar("train/label_loss", label_loss.item(), global_step)
            writer.add_scalar("train/target_loss", target_loss.item(), global_step)
            writer.add_scalar("train/distilled_loss", distilled_loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    print(f"Epoch {epoch}: Avg Label Loss: {metric_logger.meters['label_loss'].global_avg:.4f}, "
          f"Avg Target Loss: {metric_logger.meters['target_loss'].global_avg:.4f}, "
          f"Avg Distilled Loss: {metric_logger.meters['distilled_loss'].global_avg:.4f}, ")
    
    if writer is not None:
        writer.add_scalar("train/epoch_avg_label_loss", metric_logger.meters['label_loss'].global_avg, epoch)
        writer.add_scalar("train/epoch_avg_target_loss", metric_logger.meters['target_loss'].global_avg, epoch)
        writer.add_scalar("train/epoch_avg_distilled_loss", metric_logger.meters['distilled_loss'].global_avg, epoch)
    return metric_logger.meters['loss'].global_avg, iterations

def main(args):
    workspace_dir, checkpoints_dir, logs_dir, configs_dir = create_workspace(args)
    print(f"Workspace created at: {workspace_dir}")
    save_configs_and_args(args, configs_dir, args.configs)
    
    writer = SummaryWriter(logs_dir) if utils.get_rank() == 0 else None
    print(f"TensorBoard logs will be saved to: {logs_dir}")

    # -------------------------------
    # 1. 加载配置
    # -------------------------------
    configs = json.load(open(args.configs, 'r'))

    # -------------------------------
    # 2. 构建 dataset & dataloader
    # -------------------------------
    # 使用你定义的 StudentTeacherDataset
    dataset = make_object_from_config(configs["train"]["dataset"])  # 应该返回 StudentTeacherDataset 实例

    # Test dataset (for evaluation)
    dataset_test, _ = get_dataset("val", get_transform(args=args), args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # DistributedSampler
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    train_sampler = PartialDistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, fraction=configs["train"]["stream_configs"].get("data_fraction_epoch", 0.5)
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    loss_scaler = NativeScalerWithGradNormCount()
    # DataLoader with custom collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
        collate_fn=make_object_from_config(configs["train"]["collate_fn"]) # ✅ 使用 dataset 的 staticmethod collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers
    )

    print(f"Local rank {args.local_rank} | Global rank {global_rank} | "
          f"Train samples: {len(dataset)} | Val samples: {len(dataset_test)}")

    # -------------------------------
    # 3. 初始化 Student 和 Teacher 模型
    # -------------------------------
    # Student model
    model_s = segmentation.__dict__[args.model](
        pretrained=args.pretrained_swin_weights, args=args
    )
    model_s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
    model_s.cuda()
    model_s = torch.nn.parallel.DistributedDataParallel(
        model_s, device_ids=[args.local_rank], find_unused_parameters=True
    )
    single_model_s = model_s.module

    # Teacher model (same architecture, no DDP wrapper needed for EMA)
    model_t = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)  # 不加载预训练权重
    model_t = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
    model_t.cuda()
    model_t.eval()  # teacher 始终 eval 模式
    single_model_t = model_t

    # -------------------------------
    # 4. 初始化 BERT models
    # -------------------------------
    if args.model != 'lavt_one':
        bert_model_class = BertModel

        # Student BERT
        bert_s = bert_model_class.from_pretrained(args.ck_bert)
        bert_s.pooler = None
        bert_s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_s)
        bert_s.cuda()
        bert_s = torch.nn.parallel.DistributedDataParallel(
            bert_s, device_ids=[args.local_rank], find_unused_parameters=True
        )
        single_bert_s = bert_s.module

        # Teacher BERT
        bert_t = bert_model_class.from_pretrained(args.ck_bert)
        bert_t.pooler = None
        bert_t = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_t)
        bert_t.cuda()
        bert_t.eval()
        single_bert_t = bert_t

    else:
        bert_s = bert_t = single_bert_s = single_bert_t = None
    
    update_teacher_model(student=model_s, teacher=model_t, keep_rate=0)  # EMA 初始化

    # -------------------------------
    # 5. 参数分组优化
    # -------------------------------
    backbone_no_decay_s = []
    backbone_decay_s = []
    for name, m in single_model_s.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay_s.append(m)
        else:
            backbone_decay_s.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay_s, 'weight_decay': 0.0},
            {'params': backbone_decay_s},
            {"params": [p for p in single_model_s.classifier.parameters() if p.requires_grad]},
            # BERT student parameters (only first 10 layers)
            {"params": reduce(operator.concat, [
                [p for p in single_bert_s.encoder.layer[i].parameters() if p.requires_grad]
                for i in range(10)
            ])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay_s, 'weight_decay': 0.0},
            {'params': backbone_decay_s},
            {"params": [p for p in single_model_s.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat, [
                [p for p in single_model_s.text_encoder.encoder.layer[i].parameters() if p.requires_grad]
                for i in range(10)
            ])},
        ]

    # -------------------------------
    # 6. 优化器 & 学习率调度
    # -------------------------------
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    # -------------------------------
    # 7. Resume training``
    # -------------------------------
    start_epoch = 0
    best_mIoU = -0.1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        # Load student
        if "model_s" in checkpoint:
            single_model_s.load_state_dict(checkpoint['model_s'])
            if args.model != 'lavt_one':
                single_bert_s.load_state_dict(checkpoint['bert_s'])

            # Optionally load teacher (or let it be EMA-initialized)
            if 'model_t' in checkpoint:
                single_model_t.load_state_dict(checkpoint['model_t'])
            if 'bert_t' in checkpoint and args.model != 'lavt_one':
                single_bert_t.load_state_dict(checkpoint['bert_t'])
        elif "model" in checkpoint:  # 兼容旧的 checkpoint
            single_model_s.load_state_dict(checkpoint['model'])
            single_model_t.load_state_dict(checkpoint['model'])
            if args.model != 'lavt_one' and 'bert_model' in checkpoint:
                single_bert_s.load_state_dict(checkpoint['bert_model'])
                single_bert_t.load_state_dict(checkpoint['bert_model'])
                print("Loading BERT weights from checkpoint.")

        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # start_epoch = checkpoint['epoch'] + 1
        best_mIoU = checkpoint.get('best_mIoU', -0.1)
        print(f"Resumed from epoch {start_epoch}, best_mIoU: {best_mIoU:.2f}")
    

    # -------------------------------
    # 8. 损失函数 & 训练参数
    # -------------------------------
    l1 = make_object_from_config(configs["train"]["l1"])  # supervised loss
    l2 = make_object_from_config(configs["train"]["l2"])  # target loss (teacher label)
    l3 = make_object_from_config(configs["train"]["l3"])  # token consistency loss
    l_weights = configs["train"]["loss_weights"]  # [w_label, w_target, w_distill]
    keep_rate = configs["train"].get("ema_keep_rate", 0.9996)  # EMA 更新率

    # -------------------------------
    # 9. 开始训练
    # -------------------------------
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # 训练一个 epoch
        train_loss, _ = train_one_epoch(
            model_t=model_t,
            model_s=model_s,
            bert_t=bert_t,
            bert_s=bert_s,
            l1=l1,
            l2=l2,
            l3=l3,
            l_weights=l_weights,
            keep_rate=keep_rate,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            data_loader=data_loader,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            print_freq=20,
            iterations=0,  # 可扩展
            writer=writer,
            stream_configs=configs["train"]["stream_configs"]
        )

        # 评估
        iou, overallIoU = evaluate(
            model_t, data_loader_test, bert_t,
            writer=writer, epoch=epoch
        )
        print(f'Epoch {epoch}: Average object IoU {iou:.2f}, Overall IoU {overallIoU:.2f}')
        # 保存 best 模型
        save_checkpoint = (best_mIoU < iou)
        if save_checkpoint and iou > 0:
            best_mIoU = iou
            dict_to_save = {
                'model_s': single_model_s.state_dict(),
                'model_t': single_model_t.state_dict(),
                'bert_s': single_bert_s.state_dict() if bert_s is not None else None,
                'bert_t': single_bert_t.state_dict() if bert_t is not None else None,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_mIoU': best_mIoU
            }
            best_model_path = os.path.join(checkpoints_dir, f'model_best_{args.model_id}.pth')
            utils.save_on_master(dict_to_save, best_model_path)
            print(f"✅ Best model saved at: {best_model_path}")

            if writer is not None:
                writer.add_scalar("best/mIoU", best_mIoU, epoch)

        # 保存 last 模型
        dict_to_save_last = {
            'model_s': single_model_s.state_dict(),
            'model_t': single_model_t.state_dict(),
            'bert_s': single_bert_s.state_dict() if bert_s is not None else None,
            'bert_t': single_bert_t.state_dict() if bert_t is not None else None,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'best_mIoU': best_mIoU
        }
        last_model_path = os.path.join(checkpoints_dir, f'model_last_{args.model_id}.pth')
        utils.save_on_master(dict_to_save_last, last_model_path)

    # -------------------------------
    # 10. 结束 & 总结
    # -------------------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('✅ Training completed.')
    print(f"Total training time: {total_time_str}")
    print(f"Best mIoU: {best_mIoU:.2f}")

    if writer is not None:
        writer.add_text('summary/training_time', total_time_str)
        writer.add_scalar("summary/best_mIoU", best_mIoU, epoch)
        writer.close()

    
if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()


    # ----------------------- 关键修复：必须在 init_distributed_mode 之前 -----------------------
    import os
    # 从环境变量获取 LOCAL_RANK 并赋值给 args.local_rank
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # -----------------------------------------------------------------------------------

    # set up distributed learning
    # 这个函数内部会调用 torch.cuda.set_device(args.local_rank)
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)