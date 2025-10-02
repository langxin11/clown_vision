"""去噪并提取目标（小丑、气球、牵引线），背景设为黑色"""

import cv2
import numpy as np


def denoise(image):
    """对图像进行去噪处理，提取主要目标（小丑、气球、牵引线），背景设为黑色

    该函数通过高斯模糊、自适应二值化和形态学操作等步骤对图像进行去噪，
    保留主要目标（小丑、气球和牵引线），并将背景设置为黑色。

    Args:
        image (numpy.ndarray): 输入的图像，可以是彩色或灰度图像

    Returns:
        numpy.ndarray: 去噪后的二值图像，主要目标为白色，背景为黑色
    """
    # 确保输入是灰度图像
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # 1. 高斯去噪（先平滑抑制高频噪声）
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
    
    # 2. 自适应二值化（处理光照不均，保留细线条如牵引线）
    # 邻域大小11（奇数），常数C=2（控制阈值偏移）
    adaptive_binary = cv2.adaptiveThreshold(
        gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 7
    )
    #固定阈值二值化（230为阈值，255为最大像素值，THRESH_BINARY_INV为反转）
    _, adaptive_binary = cv2.threshold(gaussian_blur, 230, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 形态学操作（去除小噪点，连接断裂的牵引线）
    # 定义结构元素（3x3矩形，用于膨胀和腐蚀）
    kernel = np.ones((3, 3), np.uint8)
    # 先开运算（腐蚀+膨胀）去除小白色噪点
    opening = cv2.morphologyEx(adaptive_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 再闭运算（膨胀+腐蚀）填充目标内部小孔
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. 背景设为黑色，目标设为白色（确保输出符合要求）
    #denoised_img = cv2.bitwise_not(closing)  # 反转颜色（自适应二值化后目标为黑色，需反转）
    denoised_img = closing
    return denoised_img