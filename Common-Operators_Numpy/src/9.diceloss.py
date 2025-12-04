import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss_numpy(preds, targets, epsilon=1e-6):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    intersection = np.sum(preds_flat * targets_flat)
    union = np.sum(preds_flat) + np.sum(targets_flat)
    return 1 - (2 * intersection + epsilon) / (union + epsilon)

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6, ignore_background=False):
        """
        PyTorch Dice Loss 实现。
        可以处理二分类和多分类。
        
        参数:
        epsilon (float): 平滑项，防止除以零。
        ignore_background (bool): 是否忽略背景类别（通常是类别0）的计算。
                                  在多分类中常用。
        """
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_background = ignore_background
    def forward(self, inputs, targets):
        """
        inputs: 模型的原始输出 (Logits)。
                - 二分类: [B, 1, H, W]
                - 多分类: [B, C, H, W]
        targets: 真实标签。
                - 二分类: [B, 1, H, W] (0.0 或 1.0)
                - 多分类: [B, H, W] (类别索引 0, 1, ..., C-1)
        """
        num_classes = inputs.shape[1]
        if num_classes == 1:
            probs = torch.sigmoid(inputs)
        else:
            # 多分类
            probs = torch.softmax(inputs, dim=1)
            # 将 targets [B, H, W] (Long) F.one_hot会自动补成[B, H, W, C]
            targets = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        # --- 2. 展平，保留 Batch 和 Class 维度 ---
        # inputs/targets 形状变为 [B, C, -1]
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        # --- 3. 计算 Dice Loss ---
        intersection = (probs * targets).sum(dim=2)  # [B, C]
        union = probs.sum(dim=2) + targets.sum(dim=2)  # [B, C]
        dice_coeff = (2 * intersection + self.epsilon) / (union + self.epsilon)  # [B, C]
        
        # 先对batch维度求平均，再对class维度处理
        if num_classes == 1:
            # 二分类：对batch维度求平均
            loss = 1 - dice_coeff.mean()
        else:
            # 多分类：可以选择是否忽略背景类
            if self.ignore_background:
                # 忽略第0类（背景），只计算前景类的Dice
                loss = 1. - dice_coeff[:, 1:].mean()
            else:
                # 所有类别都参与计算
                loss = 1. - dice_coeff.mean()
        return loss

# ============ 测试示例：说明为什么不能简单地 .mean() ============
if __name__ == "__main__":
    print("="*60)
    print("测试 Dice Loss 的计算方式")
    print("="*60)
    
    # 创建测试数据
    # 假设有2个样本，3个类别
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 4, 4
    
    # 模拟logits和targets
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    
    # 测试Dice Loss
    criterion = DiceLoss(ignore_background=False)
    loss = criterion(logits, targets)
    print(f"\n多分类 Dice Loss: {loss.item():.4f}")
    
    # 展示内部计算过程
    probs = torch.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets.long(), C).permute(0, 3, 1, 2).float()
    
    probs_flat = probs.view(B, C, -1)
    targets_flat = targets_onehot.view(B, C, -1)
    
    intersection = (probs_flat * targets_flat).sum(dim=2)  # [B, C]
    union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, C]
    dice_coeff = (2 * intersection + 1e-6) / (union + 1e-6)  # [B, C]
    
    print(f"\nDice系数形状: {dice_coeff.shape}")
    print(f"Dice系数:\n{dice_coeff}")
    print(f"\n每个样本的平均Dice系数: {dice_coeff.mean(dim=1)}")
    print(f"每个类别的平均Dice系数: {dice_coeff.mean(dim=0)}")
    print(f"全局平均Dice系数: {dice_coeff.mean()}")
    
    print("\n" + "="*60)
    print("关键点：")
    print("1. dice_coeff.mean() 会同时对batch和class维度求平均")
    print("2. 这样做的问题：不同样本/类别的重要性被同等对待")
    print("3. 更好的做法：明确指定平均的维度，如 mean(dim=0) 或 mean(dim=1)")
    print("="*60)