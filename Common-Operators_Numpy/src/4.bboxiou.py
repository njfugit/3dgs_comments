import numpy as np

def calculate_iou_single(box1, box2):
    """
    计算两个边界框的IoU。

    参数:
    box1 (list or np.ndarray): 第一个边界框，格式为 [x1, y1, x2, y2]。
    box2 (list or np.ndarray): 第二个边界框，格式为 [x1, y1, x2, y2]。

    返回:
    float: 两个边界框的IoU值。
    """
    # 确保输入是numpy数组
    box1 = np.array(box1)
    box2 = np.array(box2)

    # 1. 计算交集区域的坐标
    # 左上角坐标取两者中的最大值
    x1_inter = np.maximum(box1[0], box2[0])
    y1_inter = np.maximum(box1[1], box2[1])
    # 右下角坐标取两者中的最小值
    x2_inter = np.minimum(box1[2], box2[2])
    y2_inter = np.minimum(box1[3], box2[3])

    # 2. 计算交集区域的面积
    # 确保交集区域的宽高为正数，如果不相交，则宽度或高度会为负，面积为0
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    # 3. 计算两个边界框各自的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 4. 计算并集面积
    union_area = area1 + area2 - intersection_area

    # 5. 计算IoU
    # 避免除以零的情况
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def calculate_iou_vectorized(boxes1, boxes2):
    """
    使用NumPy广播机制高效计算两组边界框的IoU矩阵。

    参数:
    boxes1 (np.ndarray): 第一组边界框，形状为 (N, 4)，格式为 [x1, y1, x2, y2]。
    boxes2 (np.ndarray): 第二组边界框，形状为 (M, 4)，格式为 [x1, y1, x2, y2]。

    返回:
    np.ndarray: IoU矩阵，形状为 (N, M)，其中iou_matrix[i, j]是boxes1[i]和boxes2[j]的IoU值。
    """
    # 计算每个边界框的面积
    # (N,)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    # (M,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 扩展维度以利用广播机制
    # boxes1: (N, 4) -> (N, 1, 4)
    # boxes2: (M, 4) -> (1, M, 4)
    boxes1_expanded = boxes1[:, np.newaxis, :]
    boxes2_expanded = boxes2[np.newaxis, :, :]

    # 1. 计算交集区域的坐标
    # np.maximum/minimum 会自动广播，结果形状为 (N, M)
    x1_inter = np.maximum(boxes1_expanded[..., 0], boxes2_expanded[..., 0])
    y1_inter = np.maximum(boxes1_expanded[..., 1], boxes2_expanded[..., 1])
    x2_inter = np.minimum(boxes1_expanded[..., 2], boxes2_expanded[..., 2])
    y2_inter = np.minimum(boxes1_expanded[..., 3], boxes2_expanded[..., 3])

    # 2. 计算交集区域的面积
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height  # 形状 (N, M)

    # 3. 计算并集面积
    # area1[:, np.newaxis] 形状 (N, 1), area2 形状 (M,)
    # 广播后相加得到 (N, M)
    union_area = area1[:, np.newaxis] + area2 - intersection_area

    # 4. 计算IoU
    # 使用 np.divide 和 where 参数处理除以零的情况
    iou_matrix = np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(intersection_area, dtype=float),
        where=union_area != 0
    )

    return iou_matrix

# --- 示例 ---
# 假设有2个预测框
pred_boxes = np.array([
    [50, 50, 150, 150],
    [300, 300, 400, 400]
])

# 假设有3个真实框
gt_boxes = np.array([
    [100, 100, 200, 200],  # 与 pred_boxes[0] 相交
    [80, 80, 120, 120],   # 与 pred_boxes[0] 相交
    [310, 320, 380, 390]   # 与 pred_boxes[1] 相交
])

iou_matrix = calculate_iou_vectorized(pred_boxes, gt_boxes)

print("批量计算的 IoU 矩阵:")
print(iou_matrix)

# 验证结果
# IoU(pred[0], gt[0])
print(f"\n验证: IoU(pred[0], gt[0]) = {calculate_iou_single(pred_boxes[0], gt_boxes[0]):.4f}")
# 矩阵中的对应值
print(f"矩阵中的值: {iou_matrix[0, 0]:.4f}")