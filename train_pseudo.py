import datetime
from torch.cuda.amp import autocast, GradScaler
import os
import time
import torch
import torch.utils.data
from torch import nn
from functools import reduce
import operator
from bert.modeling_bert import BertModel
import torchvision
from lib import segmentation
import transforms as T
import utils
import numpy as np
import torch.nn.functional as F
import gc
from collections import OrderedDict
import data.dataset_pseudo as pseudo
from loss import LabelCriterion, ConsistentDiceLoss

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


def evaluate(model, data_loader, bert_model):
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
                    output = model(image, embedding, l_mask=attentions)
                else:
                    output = model(image, sentences, l_mask=attentions)
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
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)
    return 100 * iou, 100 * cum_I / cum_U

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, lambda_consistency=1.0):
    """
    训练一个epoch的函数。

    Args:
        model: 待训练的模型。
        criterion: 原始标签的损失函数（如交叉熵）。
        optimizer: 优化器。
        data_loader: 数据加载器。
        lr_scheduler: 学习率调度器。
        epoch: 当前epoch。
        print_freq: 日志打印频率。
        iterations: 迭代次数计数器。
        bert_model: BERT模型（可选）。
        lambda_consistency: 一致性损失的权重系数。

    Returns:
        train_loss: 该epoch的平均总损失。
        iterations: 更新后的迭代次数。
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    scaler = GradScaler()
    # 添加用于记录损失的meter
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('label_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))  # 平滑显示
    metric_logger.add_meter('consistent_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    # 实例化一致性损失函数
    consistent_loss_fn = ConsistentDiceLoss(smooth=1.0)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        # 解包数据
        image, target, sentences, attentions, aug_sentences, aug_attentions = data
        # 转移到GPU
        image, target, sentences, attentions, aug_sentences, aug_attentions = (
            image.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            sentences.cuda(non_blocking=True),
            attentions.cuda(non_blocking=True),
            aug_sentences.cuda(non_blocking=True),
            aug_attentions.cuda(non_blocking=True)
        )
        # 去除多余的维度
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        aug_sentences = aug_sentences.squeeze(1)
        aug_attentions = aug_attentions.squeeze(1)
        
        with autocast():
            if bert_model is not None:
                # 使用BERT编码原始文本
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # [B, C, L]
                attentions = attentions.unsqueeze(dim=-1)  # [B, L, 1]
                image_aug = image.clone().detach() 
                # 原始输入的模型输出
                output = model(image, embedding, l_mask=attentions)

                # 使用BERT编码增强文本
                aug_last_hidden_states = bert_model(aug_sentences, attention_mask=aug_attentions)[0]
                aug_embedding = aug_last_hidden_states.permute(0, 2, 1)  # [B, C, L]
                aug_attentions = aug_attentions.unsqueeze(dim=-1).clone()  # [B, L, 1]
                
                # 关键修改：为增强输入创建 image 的副本
                # 这样，第二次模型调用不会污染第一次调用的计算图
                
                # 增强输入的模型输出
                output_aug = model(image_aug, aug_embedding.clone(), l_mask=aug_attentions.clone())
                output_aug = output_aug.detach()  # 确保增强输入的输出不影响原始输入的梯度计算
                # output_aug = torch.zeros_like(output)  # 初始化为零张量
            else:
                # 不使用BERT的情况
                output = model(image, sentences, l_mask=attentions)
                
                # 同样，为增强输入创建 image 的副本
                output_aug = model(image_aug, aug_sentences, l_mask=aug_attentions)

        # ============ 计算损失 ============
        # 1. 计算原始标签损失 (Label Loss)
        # 注意：这里仍然需要对 target 做 clone 以确保安全
        target_safe = target.clone() 
        label_loss = criterion(output, target_safe)

        # 2. 计算一致性损失 (Consistency Loss)
        # 记得禁用原始输入分支的梯度
        # consistency_loss = consistent_loss_fn(output.detach(), output_aug) 
        consistency_loss = torch.tensor(0.0, device=image.device)  # 初始化为零张量

        # 3. 计算总损失
        total_loss = label_loss + lambda_consistency * consistency_loss

        # ============ 反向传播 ============
        optimizer.zero_grad()
        # 使用scaler进行混合精度训练的反向传播
        with torch.autograd.set_detect_anomaly(True):
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ============ 更新学习率和日志 ============
        lr_scheduler.step()
        # 累加损失用于最终平均
        train_loss += total_loss.item()
        iterations += 1
        # 使用metric_logger记录各项指标
        metric_logger.update(
            loss=total_loss.item(),
            label_loss=label_loss.item(),
            consistent_loss=consistency_loss.item(),
            lr=optimizer.param_groups[0]["lr"]
        )

        # ============ 清理内存 ============
        # 删除不再需要的变量以释放GPU内存
        del image, target, sentences, attentions, aug_sentences, aug_attentions
        del output, output_aug, label_loss, consistency_loss, total_loss
        if bert_model is not None:
            del last_hidden_states, embedding, aug_last_hidden_states, aug_embedding

        # 每100次迭代清空一次缓存
        if total_its % 100 == 0:
            torch.cuda.empty_cache()

        # 同步所有进程（在分布式训练中很重要）
        torch.distributed.barrier()

    # 打印该epoch的平均损失
    print(f"Epoch {epoch}: Average Label Loss: {metric_logger.meters['label_loss'].global_avg:.4f}, "
          f"Average Consistency Loss: {metric_logger.meters['consistent_loss'].global_avg:.4f}")

    return train_loss / total_its, iterations
# 
def main(args):
    # dataset, num_classes = get_dataset("train",
    #                                    get_transform(args=args),
    #                                    args=args)
    dataset = pseudo.get_dataset(
        root="/data/datasets/tzhangbu/Cherry-Pick/data/refcoco",
        dataset="unc",
        split="train",
        max_tokens=20
    )
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    print("Building data loader...")
    print("batch size: {}".format(args.batch_size))
    print("number of workers: {}".format(args.workers))
    print("pin memory: {}".format(args.pin_mem))
    print("number of training samples: {}".format(len(dataset)))
    print("number of validation samples: {}".format(len(dataset_test)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True, persistent_workers=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    label_criterion = LabelCriterion(weight=torch.FloatTensor([0.9, 1.1]).cuda())
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, label_criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    
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