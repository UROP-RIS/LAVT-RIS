import torch
import torch.nn as nn
from torch.nn import functional as F

# class LabelCriterion(nn.Module):
#     def __init__(self, weight):
#         super().__init__()
#         self.register_buffer('weight', torch.FloatTensor(weight).cuda())

#     def forward(self, input, target):
#         return F.cross_entropy(input, target, weight=self.weight)


class LabelCriterion(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            # weight: [C], 例如 [0.5, 1.0] 表示类别 0 权重 0.5，类别 1 权重 1.0
            self.register_buffer('weight', torch.FloatTensor(weight).cuda())
        else:
            self.weight = None

    def forward(self, input, target, reduce="mean"):
        """
        input: (N, C, H, W) 或 (N, C)   -> logits
        target: (N, H, W) 或 (N,)       -> soft labels in [0.0, 1.0], float type
               对于二分类，target 是每个位置为 1 的概率
        """
        assert input.shape[1] == 2, "This is for binary classification: 2 channels (0: bg, 1: fg)"

        # 将 input 转为 log-probabilities using log_softmax
        log_prob = F.log_softmax(input, dim=1)  # (N, 2, H, W)

        # target: (N, H, W) -> expand to (N, 2, H, W)
        # 假设 target 是前景（类别 1）的概率
        # 那么类别 0 的概率就是 1 - target
        target_fg = target.unsqueeze(1)          # (N, 1, H, W)
        target_bg = 1.0 - target.unsqueeze(1)    # (N, 1, H, W)
        soft_target = torch.cat([target_bg, target_fg], dim=1)  # (N, 2, H, W)

        # 计算逐元素损失: -sum(target * log_prob)
        loss = -soft_target * log_prob  # (N, 2, H, W)

        # 加权（如果提供了类别权重）
        if self.weight is not None:
            class_weight = self.weight.view(1, 2, 1, 1)  # (1, 2, 1, 1)
            loss = loss * class_weight

        if reduce == "mean":
            return loss.sum(dim=1).mean()
        else:
            return loss.sum(dim=1)  # (N, H, W)

class LabelDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        # input: (B, 2, H, W) logits
        # 使用 sigmoid（二分类）或 softmax（多类）均可，这里用 softmax 提取前景
        prob = F.softmax(input, dim=1)
        pred = prob[:, 1]  # (B, H, W)

        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sq_sum = (pred_flat ** 2).sum(dim=1)
        target_sq_sum = (target_flat ** 2).sum(dim=1)

        dice = (2. * intersection + self.smooth) / (pred_sq_sum + target_sq_sum + self.smooth)
        loss = 1. - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class ConsistentDiceLoss(nn.Module):
    """
    一致性损失 (Consistency Loss)。
    计算原始输入和增强输入预测结果之间的一致性。
    使用 Soft IoU Loss 作为可导的损失函数。
    """
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: 平滑项，防止分母为零。
        """
        super(ConsistentDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_clean, pred_aug, scale_factor = 1.0):
        """
        Args:
            pred_clean: 原始输入的模型输出, shape [B, 2, H, W]
            pred_aug:  增强输入的模型输出, shape [B, 2, H, W]

        Returns:
            consistency_loss: 标量损失值
        """
        # 确保输入是概率分布 (应用softmax)
        # pred_clean 和 pred_aug 的形状是 [B, 2, H, W]
        # 我们沿类别维度 (dim=1) 应用 softmax，得到每个像素属于前景的概率
        prob_clean = F.softmax(pred_clean, dim=1)  # [B, 2, H, W]
        prob_aug = F.softmax(pred_aug, dim=1)      # [B, 2, H, W]

        # 我们只关心前景类别的预测概率
        # 在 RIS 任务中，通常索引 1 代表前景 (object)，索引 0 代表背景
        # 因此，我们取 softmax 后的第二个通道 (dim=1, index=1)
        foreground_clean = prob_clean[:, 1, :, :]  # [B, H, W]
        foreground_aug = prob_aug[:, 1, :, :]      # [B, H, W]

        # 将张量展平，便于计算
        # 形状从 [B, H, W] 变为 [B, H*W]
        flat_clean = foreground_clean.view(foreground_clean.size(0), -1)  # [B, N]
        flat_aug = foreground_aug.view(foreground_aug.size(0), -1)        # [B, N]

        # 计算 Soft IoU Loss
        # IoU = intersection / union
        # intersection = sum(pred * target)
        # union = sum(pred) + sum(target) - intersection
        # Soft IoU Loss = 1 - (intersection + smooth) / (union + smooth)
        
        intersection = torch.sum(flat_clean * flat_aug, dim=1)  # [B,]
        union = torch.sum(flat_clean, dim=1) + torch.sum(flat_aug, dim=1) - intersection # [B,]

        # 计算 batch 内的平均 IoU
        iou = (intersection + self.smooth) / (union + self.smooth) # [B,]
        mean_iou = torch.mean(iou) # scalar

        # 一致性损失 = 1 - 平均 Soft IoU
        # 注意：我们最小化这个损失，所以当预测越一致时，IoU 越高，损失越低。
        consistency_loss = 1.0 - mean_iou
        consistency_loss = consistency_loss * scale_factor  # 可选的缩放因子

        return consistency_loss
    
class ConsistentKLLoss(nn.Module):
    """
    一致性损失：使用 KL 散度，让学生 (pred_aug) 拟合教师 (pred_clean)
    改进：对每个像素取平均，损失不依赖分辨率
    """
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_clean, pred_aug, scale_factor=1.0):
        B, C, H, W = pred_clean.shape

        # 应用温度缩放
        log_prob_aug = F.log_softmax(pred_aug / self.temperature, dim=1)
        prob_clean = F.softmax(pred_clean / self.temperature, dim=1)

        # 计算每个像素的 KL: sum over class dimension
        # KL = sum(p_clean * (log(p_clean) - log(p_aug)))
        kl_per_pixel = torch.sum(prob_clean * (torch.log(prob_clean + 1e-8) - log_prob_aug), dim=1)  # [B, H, W]

        # 对 batch 和空间维度取平均
        kl_loss = torch.mean(kl_per_pixel)  # scalar

        # 温度补偿（Hinton et al.）
        consistency_loss = kl_loss * scale_factor * (self.temperature ** 2)

        return consistency_loss
    

if __name__ == "__main__":
    # 测试 ConsistentLoss
    pred_clean = torch.randn(2, 2, 64, 64)  # 模拟原始输入的预测
    pred_aug = torch.randn(2, 2, 64, 64)    # 模拟增强输入的预测

    criterion = ConsistentKLLoss()
    loss = criterion(pred_clean, pred_aug)
    print(f"Consistency Loss: {loss.item()}")