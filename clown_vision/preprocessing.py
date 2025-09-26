import cv2
import numpy as np

#from matplotlib import pyplot as plt

def to_gray(image):
    """将彩色图像转换为灰度图像

    使用OpenCV的cvtColor函数将BGR格式的彩色图像转换为灰度图像
    灰度化是图像处理的基础步骤，有助于后续的特征提取和分析

    Args:
        image (numpy.ndarray): 输入的BGR格式彩色图像

    Returns:
        numpy.ndarray: 转换后的单通道灰度图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_binary(gray, thresh=128):
    """将灰度图像转换为二值图像

    使用OpenCV的threshold函数对灰度图像进行全局阈值二值化处理
    大于阈值的像素点设为255（白色），小于等于阈值的像素点设为0（黑色）
    二值化有助于图像分割和特征提取，特别是在需要突出显示目标区域时

    Args:
        gray (numpy.ndarray): 输入的灰度图像
        thresh (int, optional): 二值化阈值，默认值为128

    Returns:
        numpy.ndarray: 二值化后的图像，像素值为0或255
    """
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary

def fourier_transform(gray):
    """对灰度图像进行傅里叶变换

    使用NumPy的fft2函数对灰度图像进行二维傅里叶变换
    然后使用fftshift将零频率分量移动到中心，方便分析
    最后计算幅度谱并取对数，将结果转换为8位无符号整数类型

    Args:
        gray (numpy.ndarray): 输入的灰度图像

    Returns:
        numpy.ndarray: 傅里叶变换后的幅度谱图像，像素值为0-255
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    return spectrum.astype(np.uint8)
