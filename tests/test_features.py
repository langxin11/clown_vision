import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from clown_vision.features import (
    extract_haar,
    extract_hog,
    extract_lbp,
    local_statistics,
)


def test_local_statistics_shapes_and_types():
    """测试 local_statistics 函数输出的形状、数据类型和值范围是否符合预期
    
    功能: 验证 local_statistics 函数计算的局部统计特征（均值、一阶矩、二阶矩和熵）的
    输出形状是否与输入图像一致，数据类型是否为 uint8，以及值是否在有效范围内。
    
    步骤:
    1. 创建一个随机的 64x64 8位无符号整型图像
    2. 使用窗口大小为7的 local_statistics 函数计算局部统计特征
    3. 对每个输出的统计特征数组进行以下断言:
       - 形状与输入图像相同
       - 数据类型为 uint8
       - 所有值在 0-255 范围内
    """
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)  # 创建随机测试图像
    # 计算局部统计特征: 均值、一阶矩、二阶矩和熵
    mean, moment1, moment2, entropy_img = local_statistics(img, window_size=7)
    
    # 验证每个输出特征的形状、数据类型和值范围
    for arr in [mean, moment1, moment2, entropy_img]:
        assert arr.shape == img.shape  # 确保输出形状与输入一致
        assert arr.dtype == np.uint8   # 确保数据类型为 uint8
        assert arr.max() <= 255 and arr.min() >= 0  # 确保值在有效范围内

def test_local_statistics_different_window_sizes():
    """测试 local_statistics 函数在不同窗口大小下的输出是否符合预期"""
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # 测试不同的窗口大小
    for window_size in [3, 7, 11]:
        mean, moment1, moment2, entropy_img = local_statistics(img, window_size=window_size)
        
        # 验证输出的形状和类型
        for arr in [mean, moment1, moment2, entropy_img]:
            assert arr.shape == img.shape
            assert arr.dtype == np.uint8
            assert arr.max() <= 255 and arr.min() >= 0

def test_local_statistics_color_image():
    """测试 local_statistics 函数处理彩色图像的情况"""
    # 创建一个彩色图像 (3通道)
    img_color = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # 直接传入彩色图像
    mean, moment1, moment2, entropy_img = local_statistics(img_color, window_size=7)
    
    # 验证输出
    for arr in [mean, moment1, moment2, entropy_img]:
        assert arr.shape == img_color.shape[:2]  # 应为灰度图，只有2维
        assert arr.dtype == np.uint8
        assert arr.max() <= 255 and arr.min() >= 0

def test_local_statistics_uniform_image():
    """测试 local_statistics 函数处理均匀图像的情况"""
    # 创建一个所有像素值都相同的均匀图像
    uniform_img = np.ones((64, 64), dtype=np.uint8) * 128

    mean, moment1, moment2, entropy_img = local_statistics(uniform_img, window_size=7)

    # 验证均值图像的所有像素都相同（均匀图像的局部均值应一致）
    assert np.all(mean == mean[0, 0]), "均匀图像的局部均值应一致"
    
    # 验证一阶矩接近0（因为像素值相同，绝对偏差为0）
    assert np.mean(moment1) < 1.0, "均匀图像的一阶矩应接近0"
    
    # 验证二阶矩接近0（因为像素值相同，方差为0）
    assert np.mean(moment2) < 1.0, "均匀图像的二阶矩应接近0"
    
    # 验证熵接近0（因为像素值相同，局部熵为0）
    assert np.mean(entropy_img) < 1.0, "均匀图像的熵应接近0"
    
    # 对于均匀图像，一阶矩和二阶矩应接近0（归一化后接近0）
    assert np.max(moment1) <= 20  # 归一化后应该很小
    assert np.max(moment2) <= 20  # 归一化后应该很小
    
    # 熵应接近0（归一化后接近0）
    assert np.max(entropy_img) <= 20  # 归一化后应该很小

def test_extract_lbp_basic():
    """测试 extract_lbp 函数的基本功能"""
    img = np.zeros((32, 32), dtype=np.uint8)
    lbp = extract_lbp(img)
    assert lbp.shape == img.shape
    assert lbp.dtype == np.uint8
    assert np.all(lbp == 0) or np.all(lbp <= 255)

def test_extract_lbp_different_radius():
    """测试 extract_lbp 函数在不同半径参数下的输出"""
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # 测试不同的半径值
    for radius in [1, 2, 3]:
        lbp = extract_lbp(img, radius=radius)
        assert lbp.shape == img.shape
        assert lbp.dtype == np.uint8
        assert np.max(lbp) <= 255 and np.min(lbp) >= 0

def test_extract_lbp_color_image():
    """测试 extract_lbp 函数处理彩色图像的情况"""
    # 创建一个彩色图像 (3通道)
    img_color = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # 直接传入彩色图像
    lbp = extract_lbp(img_color)
    
    # 验证输出
    assert lbp.shape == img_color.shape[:2]  # 应为灰度图，只有2维
    assert lbp.dtype == np.uint8
    assert np.max(lbp) <= 255 and np.min(lbp) >= 0

def test_extract_lbp_uniform_image():
    """测试 extract_lbp 函数处理均匀图像的情况"""
    # 创建一个所有像素值都相同的均匀图像
    uniform_img = np.ones((64, 64), dtype=np.uint8) * 128
    
    lbp = extract_lbp(uniform_img)
    
    # 对于均匀图像，LBP值应该均匀（由于是uniform LBP，相同值区域的LBP值应该相同）
    # 但由于归一化，结果会是一个中间值附近的值
    assert len(np.unique(lbp)) <= 5  # 应该只有很少的不同值

def test_extract_hog_basic():
    """测试 extract_hog 函数的基本功能"""
    img = np.ones((64, 64), dtype=np.uint8) * 128
    hog_img = extract_hog(img)
    assert hog_img.shape == img.shape
    assert hog_img.dtype == np.uint8
    assert hog_img.max() <= 255 and hog_img.min() >= 0

def test_extract_hog_different_sizes():
    """测试 extract_hog 函数处理不同大小图像的情况"""
    # 测试不同大小的图像
    sizes = [(32, 32), (64, 64), (128, 128)]
    
    for size in sizes:
        img = np.random.randint(0, 256, size, dtype=np.uint8)
        hog_img = extract_hog(img)
        
        # 验证输出
        assert hog_img.shape == img.shape
        assert hog_img.dtype == np.uint8
        assert np.max(hog_img) <= 255 and np.min(hog_img) >= 0

def test_extract_hog_color_image():
    """测试 extract_hog 函数处理彩色图像的情况"""
    # 创建一个彩色图像 (3通道)
    img_color = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # 直接传入彩色图像
    hog_img = extract_hog(img_color)
    
    # 验证输出
    assert hog_img.shape == img_color.shape[:2]  # 应为灰度图，只有2维
    assert hog_img.dtype == np.uint8
    assert np.max(hog_img) <= 255 and np.min(hog_img) >= 0

def test_extract_hog_edge_image():
    """测试 extract_hog 函数处理边缘图像的情况"""
    # 创建一个包含边缘的图像
    edge_img = np.zeros((64, 64), dtype=np.uint8)
    edge_img[:, 32:] = 255  # 垂直边缘在中间
    
    hog_img = extract_hog(edge_img)
    
    # 验证输出
    assert hog_img.shape == edge_img.shape
    assert hog_img.dtype == np.uint8
    # 边缘区域的HOG特征值应该有明显差异
    edge_area_mean = np.mean(hog_img[:, 28:36])  # 边缘附近区域
    background_mean = np.mean(hog_img[:, :10])   # 左侧背景区域
    assert abs(edge_area_mean - background_mean) > 10  # 边缘区域与背景应有明显差异

def test_extract_haar_face():
    """测试 extract_haar 函数的基本功能"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = extract_haar(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype

def test_extract_haar_different_sizes():
    """测试 extract_haar 函数处理不同大小图像的情况"""
    # 测试不同大小的图像
    sizes = [(100, 100), (200, 200), (300, 300)]
    
    for size in sizes:
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        out = extract_haar(img)
        
        # 验证输出
        assert out.shape == img.shape
        assert out.dtype == img.dtype

def test_extract_haar_with_face_like_region():
    """测试 extract_haar 函数处理可能包含人脸区域的图像"""
    # 创建一个包含类似人脸区域的图像
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # 在中间绘制一个白色矩形模拟人脸
    cv2.rectangle(img, (75, 50), (125, 120), (255, 255, 255), -1)
    
    out = extract_haar(img)
    
    # 验证输出
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    # 检查是否有检测结果（可能没有，但代码应正常运行）

if __name__ == "__main__":
    pytest.main([__file__])