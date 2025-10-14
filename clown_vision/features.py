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


def extract_haar(image):
    """提取 Haar 特征（例如检测小丑或气球等目标）

    使用 OpenCV 的 Haar 级联分类器（示例：人脸检测）

    Args:
        image (numpy.ndarray): 输入彩色图像

    Returns:
        numpy.ndarray: 绘制检测框后的输出图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    detections = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    output = image.copy()

    for (x, y, w, h) in detections:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return output
