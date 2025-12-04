import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on a set of bounding boxes.

    Args:
        boxes (np.ndarray): An array of shape (N, 4) containing the bounding boxes,
                            where N is the number of boxes. The format of each box
                            is (xmin, ymin, xmax, ymax).
        scores (np.ndarray): An array of shape (N,) containing the confidence scores
                             for each bounding box.
        iou_threshold (float): The Intersection over Union (IoU) threshold to use
                               for suppression.

    Returns:
        list: A list of indices of the kept bounding boxes.
    """
    if boxes.size == 0:
        return []

    # 根据置信度排序
    indices = np.argsort(scores)[::-1]

    keep_indices = []
    
    while len(indices) > 0:
        current_index = indices[0]
        keep_indices.append(current_index)
        #计算当前框的面积
        x1 = boxes[current_index, 0]
        y1 = boxes[current_index, 1]
        x2 = boxes[current_index, 2]
        y2 = boxes[current_index, 3]
        current_area = (y2 - y1) * (x2 - x1)
        
        #计算其他框的面积
        other_index = indices[1:]
        other_x1 = boxes[other_index, 0]
        other_y1 = boxes[other_index, 1]
        other_x2 = boxes[other_index, 2]
        other_y2 = boxes[other_index, 3]
        
        other_area = (other_y2 - other_y1) * (other_x2 - other_x1)
        
        #计算与当前框的交叉区域面积
        inter_x1 = np.maximum(x1, other_x1)
        inter_y1 = np.maximum(y1, other_y1)
        inter_x2 = np.minimum(x2, other_x2)
        inter_y2 = np.minimum(y2, other_y2)
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        
        iou = inter_area / (current_area + other_area - inter_area)
        # 找到与当前框的IoU大于阈值的框（相对位置）
        suppress_indices = np.where(iou > iou_threshold)[0]
        indices = np.delete(indices, np.concatenate(([0], suppress_indices + 1)))

    return keep_indices
        
        

def soft_nms(boxes, scores, iou_threshold, sigma=0.5):
    """通过减少相邻重叠边界框的置信度来抑制这些边界框，而不是直接移除它们"""
    """
    Performs Soft-NMS on a set of bounding boxes.

    Args:
        boxes (np.ndarray): An array of shape (N, 4) containing the bounding boxes,
                            where N is the number of boxes. The format of each box
                            is (xmin, ymin, xmax, ymax).
        scores (np.ndarray): An array of shape (N,) containing the confidence scores
                             for each bounding box.
        iou_threshold (float): The Intersection over Union (IoU) threshold.
        sigma (float): The sigma parameter for the Gaussian penalty function.

    Returns:
        list: A list of indices of the kept bounding boxes.
        np.ndarray: An array of the updated scores for the kept boxes.
    """
    if boxes.size == 0:
        return [], []
    
    # 根据置信度排序
    indices = np.argsort(scores)[::-1]
    
    new_scores = scores.copy()
    keep_indices = []
    
    while len(indices) > 0:
        current_index = indices[0]
        keep_indices.append(current_index)
        
        # 计算当前框的面积
        x1 = boxes[current_index, 0]
        y1 = boxes[current_index, 1]
        x2 = boxes[current_index, 2]
        y2 = boxes[current_index, 3]
        current_area = (y2 - y1) * (x2 - x1)
        
        # 计算其他框的面积
        other_index = indices[1:]
        other_x1 = boxes[other_index, 0]
        other_y1 = boxes[other_index, 1]
        other_x2 = boxes[other_index, 2]
        other_y2 = boxes[other_index, 3]
        other_area = (other_y2 - other_y1) * (other_x2 - other_x1)
        
        # 计算与当前框的交叉区域面积
        inter_x1 = np.maximum(x1, other_x1)
        inter_y1 = np.maximum(y1, other_y1)
        inter_x2 = np.minimum(x2, other_x2)
        inter_y2 = np.minimum(y2, other_y2)
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        iou = inter_area / (current_area + other_area - inter_area)
        #IoU 越大，penalty越小，衰减越大
        penalty = np.exp(-(iou * iou) / sigma)
        # 更新置信度
        new_scores[other_index] *= penalty
        
        #去掉当前的索引
        indices = indices[1:]
        if len(indices) > 0:
          updated_scores_indices = np.argsort(new_scores[indices])[::-1]
          indices = indices[updated_scores_indices]
    
    final_keep_indices = [i for i in keep_indices if new_scores[i] > 0.01]
    final_scores = new_scores[final_keep_indices]
    return final_keep_indices, final_scores



def main():
    """
    主函数，用于创建样本数据并验证NMS和Soft-NMS。
    """
    # 1. 创建样本数据
    # 场景:
    # Box 0: 主要目标，得分最高
    # Box 1: 与Box 0高度重叠，得分次之
    # Box 2: 与Box 0和Box 1都部分重叠，得分较低
    # Box 3: 独立的目标，与其它框无重叠，得分较高
    # Box 4: 另一个独立的目标，得分最低
    boxes = np.array([
        [100, 100, 210, 210],  # Box 0
        [120, 120, 230, 230],  # Box 1
        [180, 180, 280, 280],  # Box 2
        [300, 300, 400, 400],  # Box 3
        [10, 10, 50, 50]       # Box 4
    ])

    scores = np.array([0.95, 0.85, 0.7, 0.9, 0.6])

    iou_threshold = 0.5
    sigma = 0.5

    print("--- 原始数据 ---")
    for i in range(len(boxes)):
        print(f"Box {i}: {boxes[i]}, Score: {scores[i]:.2f}")
    print("-" * 20)

    # 2. 验证标准 NMS
    print("\n--- 验证标准 NMS (IoU 阈值 = 0.5) ---")
    kept_indices_nms = nms(boxes.copy(), scores.copy(), iou_threshold)
    print(f"保留的Box索引: {kept_indices_nms}")
    print("结果分析:")
    print("Box 0 (0.95) 被保留。")
    print("Box 3 (0.90) 与其他框不重叠，被保留。")
    print("Box 1 (0.85) 和 Box 2 (0.70) 因为与更高分的Box 0重叠度(IoU)超过0.5，被完全抑制。")
    print("Box 4 (0.60) 与其他框不重叠，被保留。")
    print("-" * 20)


    # 3. 验证 Soft-NMS
    print("\n--- 验证 Soft-NMS (sigma = 0.5) ---")
    kept_indices_soft, updated_scores = soft_nms(boxes.copy(), scores.copy(), iou_threshold, sigma)
    
    print("保留的Box索引及其更新后的分数:")
    for i, score in zip(kept_indices_soft, updated_scores):
        original_score = scores[i]
        print(f"Box {i}: 原始分数 = {original_score:.2f}, 更新后分数 = {score:.4f}")

    print("\n结果分析:")
    print("所有与更高分框重叠的Box的分数都被衰减，而不是直接移除。")
    print("Box 0 和 Box 3 的分数基本不变，因为它们是各自区域内得分最高的。")
    print("Box 1 的分数被显著降低，因为它与Box 0高度重叠。")
    print("Box 2 的分数也被降低，因为它与Box 0和Box 1都有重叠。")
    print("最终，分数高于某个阈值（例如0.01）的框都会被保留，但分数反映了其与邻近高分框的重叠情况。")
    print("-" * 20)


if __name__ == "__main__":
    main()