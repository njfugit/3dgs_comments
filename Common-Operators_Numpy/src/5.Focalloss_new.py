# focalloss: 不同的权重来调整 loss
#

import numpy as np

def binary_focal_loss(y_true, y_pred, alpha = 0.5, gamma = 2):
    """
    计算二分类问题的 Focal Loss。
    focal loss = -αt⋅(1−p_t)^γ⋅log(p_t)
    参数:
    y_true: numpy 数组, 真实的标签 (0 或 1)。形状为 (N,)。
    y_pred: numpy 数组, 模型预测为类别 1 的概率 (通常来自 Sigmoid 激活)。
            形状应与 y_true 相同。
    alpha: float, 平衡正负样本权重的超参数。通常设为少数类别的权重。
    gamma: float, 聚焦参数，用于调节简单样本权重降低的速率。
    epsilon: float, 一个极小值，用于防止 log(0) 出现计算错误。

    返回:
    float, 计算出的 Focal Loss 值的平均值。
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    #p_t=y⋅p+(1−y)⋅(1−p)，是模型对于正确类别的预测概率  y是真实标签 (0 或 1)，p 是模型预测为 1 的概率
    #p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    
    # 计算 log(p_t)
    log_p_t = np.log(p_t)
    
    #αt=y⋅α+(1−y)⋅(1−α)，是用于平衡正负样本的权重
    # alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    # 计算 (1 - p_t)^gamma (调制因子)
    modulating_factor = (1 - p_t) ** gamma
    
    # 计算 focal loss = -αt⋅(1−p_t)^γ⋅log(p_t)
    focal_loss = -alpha_t * modulating_factor * log_p_t
    
    # 返回平均 focal loss
    return np.mean(focal_loss)

def categorical_focal_loss(y_true, y_pred, alpha, gamma=2.0, epsilon=1e-12):
    """
    计算多类别问题的 Focal Loss。
    这里的数据格式有所不同，y_true是one-hot编码，利用 One-Hot 编码的特性，y_true 中只有一个 1，其余都是 0
    参数:
    y_true: numpy 数组, 真实的 one-hot 编码标签。形状为 (N, C)。
    y_pred: numpy 数组, 模型预测的概率 (通常来自 Softmax 激活)。
            形状应与 y_true 相同 (N, C)。
    alpha: numpy 数组, 各类别的权重。形状为 (C,) 或 (1, C)。
           用于平衡不同类别的重要性。
    gamma: float, 聚焦参数。
    epsilon: float, 防止 log(0) 的小值。

    返回:
    float, 计算出的 Focal Loss 值的平均值。
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    # 我们用 y_true 作为掩码来选择正确类别的预测概率
    p_t = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    # 2. 计算 log(p_t)
    log_p_t = np.log(p_t)
    # 计算 (1 - p_t)^gamma (调制因子)
    modulating_factor = (1 - p_t) ** gamma
    # 3. alpha 权重也需要通过 y_true 来选择
    alpha_t = np.sum(y_true * alpha, axis=-1, keepdims=True)
    # 4. 计算 focal loss = -αt⋅(1−p_t)^γ⋅log(p_t)
    focal_loss = -alpha_t * modulating_factor * log_p_t
    return np.mean(focal_loss)

if __name__ == "__main__":
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])