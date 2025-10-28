"""
测试增强的牵引绳分色算法
主要测试改进的交叉点处理算法效果
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from clown_vision.denoising import denoise
from clown_vision.rescue import ClownRescuer

# 导入需要测试的模块
from clown_vision.untangle import (
    _analyze_line_geometry,
    _detect_intersections,
    _global_topology_optimization,
    _smart_multisource_bfs_label,
    colorize_lines,
    process_pipeline,
)


class TestEnhancedUntangle:
    """测试增强的牵引绳分色算法"""
    
    @pytest.fixture
    def test_image(self):
        """加载测试图像"""
        # 使用项目中的测试图像
        test_img_path = Path(__file__).parent.parent / "assets" / "test.png"
        if test_img_path.exists():
            return cv2.imread(str(test_img_path))
        else:
            # 如果测试图像不存在，创建一个简单的测试图像
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            # 绘制三条交叉的线
            cv2.line(img, (50, 50), (250, 250), (255, 255, 255), 3)
            cv2.line(img, (50, 250), (250, 50), (255, 255, 255), 3)
            cv2.line(img, (150, 50), (150, 250), (255, 255, 255), 3)
            return img
    
    @pytest.fixture
    def skeleton_image(self, test_image):
        """生成骨架图像"""
        # 去噪
        denoised = denoise(test_image)
        # 生成骨架
        from skimage.morphology import skeletonize
        skel = skeletonize((denoised > 0)).astype(np.uint8)
        return skel
    
    def test_enhanced_intersection_detection(self, skeleton_image):
        """测试增强的交叉点检测算法"""
        # 检测交叉点
        intersections = _detect_intersections(skeleton_image)
        
        # 验证检测结果
        assert isinstance(intersections, list)
        assert len(intersections) > 0
        
        # 验证交叉点坐标在图像范围内
        H, W = skeleton_image.shape
        for y, x in intersections:
            assert 0 <= y < H
            assert 0 <= x < W
            assert skeleton_image[y, x] > 0  # 交叉点应该在骨架上
        
        print(f"检测到 {len(intersections)} 个交叉点")
    
    def test_line_geometry_analysis(self, skeleton_image):
        """测试线段几何特征分析"""
        # 选择一个方向和起点
        direction = (1, 0)  # 向右
        point = (150, 100)  # 起点
        
        # 分析线段几何特征
        curvature, direction_consistency, has_branch = _analyze_line_geometry(
            skeleton_image, point, direction, length=10
        )
        
        # 验证结果
        assert isinstance(curvature, float)
        assert isinstance(direction_consistency, float)
        assert isinstance(has_branch, bool)
        assert 0 <= direction_consistency <= 1
        
        print(f"曲率: {curvature:.4f}, 方向一致性: {direction_consistency:.4f}, 有分叉: {has_branch}")
    
    def test_global_topology_optimization(self, skeleton_image):
        """测试全局拓扑优化"""
        # 创建一个简单的标签图
        H, W = skeleton_image.shape
        labels = np.zeros((H, W), dtype=np.uint8)
        
        # 手动标记一些区域
        labels[100:150, 100:150] = 1
        labels[150:200, 150:200] = 2
        labels[100:150, 150:200] = 3
        
        # 检测交叉点
        intersections = _detect_intersections(skeleton_image)
        
        # 应用全局拓扑优化
        optimized_labels = _global_topology_optimization(labels, skeleton_image, intersections)
        
        # 验证结果
        assert optimized_labels.shape == labels.shape
        assert np.all(optimized_labels >= 0)
        
        print(f"优化前标签数: {len(np.unique(labels))}, 优化后标签数: {len(np.unique(optimized_labels))}")
    
    def test_enhanced_multisource_bfs(self, skeleton_image):
        """测试增强的多源BFS算法"""
        # 选择种子点
        seeds = [(100, 100), (200, 200), (150, 50)]
        
        # 应用增强的多源BFS
        label_map = _smart_multisource_bfs_label(skeleton_image, seeds)
        
        # 验证结果
        assert label_map.shape == skeleton_image.shape
        assert np.all(label_map >= 0)
        
        # 验证种子点被正确标记
        for i, (y, x) in enumerate(seeds, start=1):
            if 0 <= y < skeleton_image.shape[0] and 0 <= x < skeleton_image.shape[1]:
                if skeleton_image[y, x] > 0:
                    assert label_map[y, x] == i
        
        print(f"标签分配完成，标签范围: {np.unique(label_map)}")
    
    def test_colorize_lines_with_enhanced_algorithm(self, test_image):
        """测试使用增强算法的牵引绳分色"""
        # 指定种子点
        seed_points = [(100, 100), (200, 200), (150, 50)]
        
        # 应用增强的分色算法
        result = colorize_lines(
            test_image,
            seed_points_xy=seed_points,
            interactive_mode=False
        )
        
        # 验证结果
        assert result is not None
        assert result.shape == test_image.shape
        
        # 检查结果中是否有颜色
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) > 1  # 应该有多种颜色
        
        print(f"分色完成，检测到 {len(unique_colors)} 种颜色")
    
    def test_full_pipeline_with_enhanced_algorithm(self, test_image):
        """测试使用增强算法的完整处理流程"""
        # 应用完整处理流程
        result = process_pipeline(test_image, interactive_seed_selection=False)
        
        # 验证结果
        assert result is not None
        assert result.shape == test_image.shape
        
        # 检查结果中是否有颜色
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) > 1  # 应该有多种颜色
        
        print(f"完整处理流程完成，检测到 {len(unique_colors)} 种颜色")
    
    def test_performance_comparison(self, test_image):
        """测试增强算法的性能改进"""
        import time
        
        # 测试原始算法（通过模拟）
        start_time = time.time()
        # 这里我们无法直接测试原始算法，但可以测试增强算法的性能
        result_enhanced = process_pipeline(test_image, interactive_seed_selection=False)
        enhanced_time = time.time() - start_time
        
        # 验证结果质量
        assert result_enhanced is not None
        
        # 检查处理时间是否合理
        assert enhanced_time < 30.0  # 处理时间不应超过30秒
        
        print(f"增强算法处理时间: {enhanced_time:.2f}秒")
        
        # 分析结果质量
        unique_colors = np.unique(result_enhanced.reshape(-1, 3), axis=0)
        print(f"检测结果包含 {len(unique_colors)} 种颜色")
        
        # 检查是否有明显的三种主要颜色（红、绿、蓝）
        red_count = np.sum(np.all(result_enhanced == [0, 0, 255], axis=2))
        green_count = np.sum(np.all(result_enhanced == [0, 255, 0], axis=2))
        blue_count = np.sum(np.all(result_enhanced == [255, 0, 0], axis=2))
        
        total_colored = red_count + green_count + blue_count
        total_pixels = result_enhanced.shape[0] * result_enhanced.shape[1]
        
        print(f"红色像素: {red_count}, 绿色像素: {green_count}, 蓝色像素: {blue_count}")
        print(f"着色覆盖率: {total_colored / total_pixels * 100:.2f}%")
        
        # 着色覆盖率应该合理
        assert total_colored / total_pixels > 0.01  # 至少1%的像素被着色
    
    def test_intersection_handling_robustness(self):
        """测试交叉点处理的鲁棒性"""
        # 创建一个复杂的交叉点测试图像
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # 创建多个交叉点
        cv2.line(img, (100, 100), (300, 300), (255, 255, 255), 3)
        cv2.line(img, (100, 300), (300, 100), (255, 255, 255), 3)
        cv2.line(img, (200, 50), (200, 350), (255, 255, 255), 3)
        cv2.line(img, (50, 200), (350, 200), (255, 255, 255), 3)
        
        # 生成骨架
        denoised = denoise(img)
        from skimage.morphology import skeletonize
        skel = skeletonize((denoised > 0)).astype(np.uint8)
        
        # 检测交叉点
        intersections = _detect_intersections(skel)
        
        # 验证检测到的交叉点数量
        assert len(intersections) >= 1  # 至少应该检测到一个交叉点
        
        # 测试多源BFS在复杂交叉点情况下的表现
        seeds = [(100, 100), (300, 300), (200, 50)]
        label_map = _smart_multisource_bfs_label(skel, seeds)
        
        # 验证标签分配
        assert np.max(label_map) <= 3  # 最多3个标签
        assert np.min(label_map) >= 0
        
        print(f"复杂交叉点测试：检测到 {len(intersections)} 个交叉点，标签分配完成")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])