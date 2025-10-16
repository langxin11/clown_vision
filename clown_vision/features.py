"""
提供图像局部均值、方差、信息熵等统计特征的计算，
以及纹理和形状特征（LBP、HOG、Haar）的提取。
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.measure import shannon_entropy
from skimage import exposure
from skimage.filters.rank import entropy
from skimage.morphology import disk


def local_statistics(gray, window_size=7):
    # 统一到灰度与类型
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g_u8 = gray.astype(np.uint8)         # rank.entropy 需要 8-bit 图  【熵用】
    g = gray.astype(np.float32)

    # 均值 μ = E[I]
    k = (window_size, window_size)
    local_mean = cv2.blur(g, k)

    # 一阶“中心绝对矩”：E[|I-μ|] —— 稳健度更好，且不会恒为0
    abs_dev = cv2.absdiff(g, local_mean)
    moment1 = cv2.blur(abs_dev, k)

    # 二阶中心矩：E[(I-μ)^2] —— 即方差（未开方）
    sq_dev = (g - local_mean) ** 2
    moment2 = cv2.blur(sq_dev, k)

    # 信息熵：基于 rank.entropy 的局部熵（结构元素用半径 window_size//2 的圆盘）
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    entropy_img = entropy(g_u8, disk(max(1, window_size // 2))).astype(np.float32)

    # 统一归一化到 0–255 并转 uint8
    def norm8(x):
        return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return norm8(local_mean), norm8(moment1), norm8(moment2), norm8(entropy_img)



def extract_lbp(gray, radius=1):
    """提取 LBP（局部二值模式）纹理特征

    Args:
        gray (numpy.ndarray): 输入灰度图像
        radius (int): 邻域半径（默认1）

    Returns:
        numpy.ndarray: LBP 特征图
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    return (lbp / np.max(lbp) * 255).astype(np.uint8)


def extract_hog(gray):
    """提取 HOG（梯度方向直方图）特征

    Args:
        gray (numpy.ndarray): 输入灰度图像

    Returns:
        numpy.ndarray: 可视化的 HOG 特征图
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return (hog_image * 255).astype(np.uint8)


def extract_haar(image,
                 cascades=('haarcascade_frontalface_alt2.xml',
                           'haarcascade_frontalface_default.xml',
                           'haarcascade_frontalface_alt.xml',
                           'haarcascade_profileface.xml'),
                 scale_factor=1.05,      # ⬅ 召回↑：1.05；更稳：1.1
                 min_neighbors=3,        # ⬅ 召回↑：降到2/3；更稳：5
                 min_size_ratio=0.05,    # ⬅ 最小目标尺寸占短边比例，调 0.04~0.10
                 max_size_ratio=0.60,    # ⬅ 最大目标尺寸占短边比例
                 use_clahe=True,         # ⬅ 弱纹理图建议开
                 use_bilateral=True,     # ⬅ 去噪保边
                 second_chance=True):    # ⬅ 第一次没检出时，放宽再试一轮
    """
    return: 画好框的 BGR 图（无检出则给黄字提示）
    """
    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # —— 轻量预处理（可关）
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # —— 加载可用模型
    cds = []
    root = cv2.data.haarcascades
    for name in cascades:
        clf = cv2.CascadeClassifier(root + name)
        if not clf.empty():
            cds.append(clf)
    if not cds:
        cv2.putText(out, 'Haar cascade not found', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        return out

    H, W = gray.shape
    smin = max(16, int(min(H, W) * min_size_ratio))
    smax = int(min(H, W) * max_size_ratio)

    def detect_once(sf, mn):
        rects = []
        for clf in cds:
            det = clf.detectMultiScale(gray,
                                       scaleFactor=sf,
                                       minNeighbors=mn,
                                       flags=cv2.CASCADE_SCALE_IMAGE,
                                       minSize=(smin, smin),
                                       maxSize=(smax, smax))
            if det is not None and len(det) > 0:
                rects.extend(det.tolist())
        return rects

    # —— 第一次尝试：用户给定阈值
    rects = detect_once(scale_factor, min_neighbors)

    # —— 二次放宽（可关）
    if second_chance and len(rects) == 0:
        rects = detect_once(max(1.03, scale_factor*0.98), max(2, min_neighbors-1))

    if len(rects) == 0:
        cv2.putText(out, 'No Haar detections', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
        return out

    # —— 极简 NMS，避免多层级联重复框
    boxes = np.array(rects, dtype=np.float32)
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1); h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter/(areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= 0.3)[0] + 1]
    rects = boxes[keep].astype(int).tolist()

    for (x, y, w, h) in rects:
        cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)
    return out
