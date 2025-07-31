import torch
import torch.nn as nn
from torch.nn import functional as F

class LabelCriterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight)

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
        pred_clean = pred_clean.detach()
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
    一致性损失 (Consistency Loss)。
    使用 KL 散度 (KL Divergence) 作为损失函数。
    让增强输入的预测 (student) 拟合原始输入的预测 (teacher)。
    """
    def __init__(self, temperature=1.0, scale_factor=1.0):
        """
        Args:
            temperature: 温度系数，用于软化概率分布。
                         温度越高，分布越平滑；温度越低，分布越尖锐。
            scale_factor: 损失的缩放因子。
        """
        super(ConsistentKLLoss, self).__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        # 使用 nn.KLDivLoss，注意其第一个参数是 log-probabilities
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred_clean, pred_aug):
        """
        Args:
            pred_clean: 原始输入的模型输出 (logits), shape [B, 2, H, W]
            pred_aug:   增强输入的模型输出 (logits), shape [B, 2, H, W]

        Returns:
            consistency_loss: 标量损失值
        """
        # 获取张量的形状
        B, C, H, W = pred_clean.shape
        
        # 步骤1: 将 logits 转换为 log-probabilities (用于KL散度的输入)
        # 注意：F.log_softmax 是计算 log-probabilities 的标准方法
        log_prob_aug = F.log_softmax(pred_aug / self.temperature, dim=1)  # [B, 2, H, W]
        
        # 步骤2: 将 logits 转换为 probabilities (用于KL散度的目标)
        # 注意：F.softmax 是计算 probabilities 的标准方法
        prob_clean = F.softmax(pred_clean / self.temperature, dim=1)       # [B, 2, H, W]
        
        # 步骤3: 计算KL散度
        # KL(P_clean || P_aug) = sum(P_clean * log(P_clean / P_aug))
        # nn.KLDivLoss 计算的是 sum(target * (log_target - log_input))，所以:
        # input = log_prob_aug, target = prob_clean
        # 注意：我们这里使用的是 "reverse KL"，即用 P_aug 去拟合 P_clean。
        # 这是自蒸馏中的标准做法：让学生 (aug) 拟合教师 (clean)。
        kl_div_loss = self.kl_loss(log_prob_aug, prob_clean)  # 标量
        
        # 应用缩放因子
        consistency_loss = kl_div_loss * self.scale_factor
        
        return consistency_loss
    

if __name__ == "__main__":
    # 测试 ConsistentLoss
    pred_clean = torch.randn(2, 2, 64, 64)  # 模拟原始输入的预测
    pred_aug = torch.randn(2, 2, 64, 64)    # 模拟增强输入的预测

    criterion = ConsistentKLLoss()
    loss = criterion(pred_clean, pred_aug)
    print(f"Consistency Loss: {loss.item()}")