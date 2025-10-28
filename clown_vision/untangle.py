"""
牵引绳分色模块（重构版）
功能：利用任务2去噪和任务5解救小丑的结果，通过人为选取三个目标线的种子点，
      在交叉点基于方向一致性选择路径，实现牵引绳分色
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import denoising  # 复用任务2的掩膜（黑底白前景）
from .rescue import ClownRescuer  # 复用任务5的结果


# ---------- 工具函数 ----------
def _nearest_on_skeleton(y: int, x: int, skel: np.ndarray, max_radius: int = 60):
    """从(y,x)出发在递增半径内寻找最近的骨架像素；找不到返回 None。"""
    H, W = skel.shape
    for r in range(1, max_radius + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        yy, xx = np.nonzero(skel[y0:y1, x0:x1])
        if len(yy) > 0:
            yy, xx = yy + y0, xx + x0
            idx = np.argmin((yy - y) ** 2 + (xx - x) ** 2)
            return int(yy[idx]), int(xx[idx])
    return None


def _get_direction_vector(prev_pos: tuple, curr_pos: tuple) -> tuple:
    """计算从prev_pos到curr_pos的方向向量"""
    dy = curr_pos[0] - prev_pos[0]
    dx = curr_pos[1] - prev_pos[1]
    return (dy, dx)


def _angle_between_vectors(vec1: tuple, vec2: tuple) -> float:
    """计算两个向量之间的夹角（弧度）"""
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    # 归一化
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return np.pi
    
    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    
    # 计算点积，得到夹角的余弦值
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle


def _detect_intersections(skel: np.ndarray):
    """
    增强的交叉点检测算法：
    1. 基于度检测（度>=4的点）
    2. 基于局部拓扑结构验证
    3. 基于几何特征分析
    """
    H, W = skel.shape
    
    # 1. 基于度检测
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], np.uint8)
    deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    candidates = np.where(deg >= 14)  # 度>=4的候选点
    
    # 2. 局部拓扑结构验证
    intersections = []
    for y, x in zip(candidates[0], candidates[1]):
        # 提取局部区域
        y0, y1 = max(0, y-2), min(H, y+3)
        x0, x1 = max(0, x-2), min(W, x+3)
        local_region = skel[y0:y1, x0:x1]
        
        # 计算局部区域的连通分量数
        num_labels, labels = cv2.connectedComponents(local_region.astype(np.uint8), connectivity=8)
        
        # 如果连通分量数大于等于3，认为是真正的交叉点
        if num_labels >= 3:
            intersections.append((y, x))
    
    # 3. 几何特征分析 - 移除过于接近的交叉点
    filtered_intersections = []
    min_distance = 10  # 最小距离阈值
    
    for y, x in intersections:
        too_close = False
        for fy, fx in filtered_intersections:
            if np.sqrt((y-fy)**2 + (x-fx)**2) < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered_intersections.append((y, x))
    
    return filtered_intersections


def _analyze_line_geometry(skel: np.ndarray, point: tuple, direction: tuple, length: int = 10):
    """
    分析线段的几何特征：
    1. 计算线段的曲率
    2. 分析线段的方向一致性
    3. 检测线段的分叉情况
    
    Args:
        skel: 骨架图像
        point: 起始点 (y, x)
        direction: 初始方向 (dy, dx)
        length: 分析长度
    
    Returns:
        curvature: 曲率
        direction_consistency: 方向一致性 (0-1)
        has_branch: 是否有分叉
    """
    y, x = point
    dy, dx = direction
    
    # 归一化方向向量
    norm = np.sqrt(dy**2 + dx**2)
    if norm > 0:
        dy, dx = dy/norm, dx/norm
    
    # 跟踪线段
    points = [(y, x)]
    directions = [(dy, dx)]
    
    for _ in range(length):
        # 在当前点周围寻找下一个骨架点
        next_points = []
        for ndy, ndx in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            ny, nx = y + ndy, x + ndx
            if (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and
                skel[ny, nx] and (ny, nx) not in points):
                next_points.append((ny, nx, ndy, ndx))
        
        if not next_points:
            break
        
        # 选择与当前方向最一致的点
        best_point = None
        best_alignment = -1
        
        for ny, nx, ndy, ndx in next_points:
            # 归一化新方向
            new_norm = np.sqrt(ndy**2 + ndx**2)
            if new_norm > 0:
                ndy, ndx = ndy/new_norm, ndx/new_norm
            
            # 计算与当前方向的一致性
            alignment = dy*ndy + dx*ndx
            if alignment > best_alignment:
                best_alignment = alignment
                best_point = (ny, nx, ndy, ndx)
        
        if best_point is None:
            break
        
        y, x, dy, dx = best_point
        points.append((y, x))
        directions.append((dy, dx))
    
    # 计算几何特征
    if len(points) < 3:
        return 0.0, 1.0, False
    
    # 1. 计算曲率
    curvature = 0.0
    for i in range(1, len(points)-1):
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        p3 = np.array(points[i+1])
        
        # 计算向量
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算夹角
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            curvature += angle
    
    curvature /= (len(points) - 2)
    
    # 2. 计算方向一致性
    direction_consistency = 0.0
    if len(directions) > 1:
        initial_dir = np.array(directions[0])
        for dir_vec in directions[1:]:
            dir_vec = np.array(dir_vec)
            consistency = np.dot(initial_dir, dir_vec)
            direction_consistency += consistency
        
        direction_consistency /= len(directions)
        direction_consistency = (direction_consistency + 1) / 2  # 归一化到0-1
    
    # 3. 检测分叉
    has_branch = False
    for y, x in points:
        # 检查每个点的邻域
        neighbor_count = 0
        for ndy, ndx in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            ny, nx = y + ndy, x + ndx
            if (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and
                skel[ny, nx]):
                neighbor_count += 1
        
        if neighbor_count > 3:  # 除了当前路径，还有其他分支
            has_branch = True
            break
    
    return curvature, direction_consistency, has_branch


def _global_topology_optimization(labels: np.ndarray, skel: np.ndarray, intersections: list):
    """
    全局拓扑优化：
    1. 分析整体拓扑结构
    2. 优化交叉点处的标签分配
    3. 确保路径的连续性和一致性
    
    Args:
        labels: 标签图像
        skel: 骨架图像
        intersections: 交叉点列表
    
    Returns:
        优化后的标签图像
    """
    H, W = labels.shape
    optimized_labels = labels.copy()
    
    # 1. 为每个交叉点分析连接的线段
    for y, x in intersections:
        # 找到所有连接到该交叉点的线段
        connected_segments = {}
        
        for dy, dx in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < H and 0 <= nx < W and skel[ny, nx]):
                # 跟踪线段
                segment_points = []
                current_y, current_x = ny, nx
                segment_label = labels[current_y, current_x]
                
                # 沿着线段跟踪，直到到达另一个交叉点或端点
                while True:
                    segment_points.append((current_y, current_x))
                    
                    # 检查是否到达另一个交叉点
                    is_intersection = False
                    for iy, ix in intersections:
                        if (current_y, current_x) == (iy, ix):
                            is_intersection = True
                            break
                    
                    if is_intersection:
                        break
                    
                    # 检查是否为端点
                    neighbor_count = 0
                    for ndy, ndx in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        nny, nnx = current_y + ndy, current_x + ndx
                        if (0 <= nny < H and 0 <= nnx < W and skel[nny, nnx]):
                            neighbor_count += 1
                    
                    if neighbor_count == 1:  # 端点
                        break
                    
                    # 移动到下一个点
                    moved = False
                    for ndy, ndx in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        nny, nnx = current_y + ndy, current_x + ndx
                        if (0 <= nny < H and 0 <= nnx < W and skel[nny, nnx] and
                            (nny, nnx) not in segment_points):
                            current_y, current_x = nny, nnx
                            moved = True
                            break
                    
                    if not moved:
                        break
                
                # 记录线段信息
                if segment_label > 0:
                    if segment_label not in connected_segments:
                        connected_segments[segment_label] = []
                    connected_segments[segment_label].append(segment_points)
        
        # 2. 优化交叉点处的标签分配
        if len(connected_segments) > 1:
            # 分析每个标签的线段特征
            segment_features = {}
            for label, segments in connected_segments.items():
                total_length = sum(len(seg) for seg in segments)
                avg_curvature = 0.0
                avg_direction_consistency = 0.0
                
                for segment in segments:
                    if len(segment) >= 3:
                        # 计算线段的几何特征
                        points_array = np.array(segment)
                        # 简化的曲率计算
                        vectors = np.diff(points_array, axis=0)
                        if len(vectors) > 1:
                            angles = []
                            for i in range(len(vectors)-1):
                                v1 = vectors[i]
                                v2 = vectors[i+1]
                                norm1 = np.linalg.norm(v1)
                                norm2 = np.linalg.norm(v2)
                                if norm1 > 0 and norm2 > 0:
                                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                                    angles.append(np.arccos(cos_angle))
                            
                            if angles:
                                avg_curvature += np.mean(angles)
                        
                        # 简化的方向一致性计算
                        if len(vectors) > 0:
                            main_direction = vectors[0]
                            consistency_sum = 0
                            for v in vectors:
                                norm_v = np.linalg.norm(v)
                                if norm_v > 0:
                                    consistency_sum += np.dot(main_direction, v) / norm_v
                            
                            avg_direction_consistency += consistency_sum / len(vectors)
                
                if len(segments) > 0:
                    avg_curvature /= len(segments)
                    avg_direction_consistency = (avg_direction_consistency / len(segments) + 1) / 2  # 归一化到0-1
                
                segment_features[label] = {
                    'total_length': total_length,
                    'avg_curvature': avg_curvature,
                    'avg_direction_consistency': avg_direction_consistency,
                    'segment_count': len(segments)
                }
            
            # 3. 选择最优标签
            best_label = None
            best_score = -1
            
            for label, features in segment_features.items():
                # 综合评分：长度权重0.4，方向一致性权重0.4，曲率权重0.2
                length_score = min(features['total_length'] / 50.0, 1.0)  # 归一化长度
                direction_score = features['avg_direction_consistency']
                curvature_score = 1.0 - min(features['avg_curvature'] / np.pi, 1.0)  # 曲率越小越好
                
                total_score = 0.4 * length_score + 0.4 * direction_score + 0.2 * curvature_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_label = label
            
            # 4. 应用优化结果
            if best_label is not None:
                optimized_labels[y, x] = best_label
                
                # 优化交叉点周围的标签
                for dy, dx in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W and skel[ny, nx]):
                        # 如果该点原本没有标签或者标签不一致，使用最优标签
                        if optimized_labels[ny, nx] == 0 or optimized_labels[ny, nx] != best_label:
                            optimized_labels[ny, nx] = best_label
    
    return optimized_labels


def _smart_multisource_bfs_label(skel: np.ndarray, seeds):
    """
    增强的多源BFS算法，专门处理交叉点：
    - 在交叉点处基于方向一致性和几何特征选择路径
    - 使用线段几何特征分析优化路径选择
    - 避免在交叉点处错误分配标签
    """
    H, W = skel.shape
    label_map = np.zeros((H, W), np.uint8)
    dist = np.full((H, W), 1_000_000, np.int32)
    
    # 记录每个点的路径历史，用于方向一致性判断
    path_history = {}
    
    # 检测交叉点
    intersections = _detect_intersections(skel)
    
    # 初始化种子点
    q = []
    for i, (y, x) in enumerate(seeds, start=1):
        if 0 <= y < H and 0 <= x < W and skel[y, x]:
            label_map[y, x] = i
            dist[y, x] = 0
            path_history[(y, x)] = [(y, x)]  # 记录路径
            q.append((y, x))

    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))

    head = 0
    while head < len(q):
        y, x = q[head]; head += 1
        
        # 获取当前点的路径历史
        current_path = path_history.get((y, x), [])
        
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                nd = dist[y, x] + 1
                
                # 如果找到更短的路径
                if nd < dist[ny, nx]:
                    # 检查是否为交叉点
                    is_intersection = (ny, nx) in intersections
                    
                    if not is_intersection:
                        # 非交叉点，直接分配标签
                        dist[ny, nx] = nd
                        label_map[ny, nx] = label_map[y, x]
                        path_history[(ny, nx)] = current_path + [(ny, nx)]
                        q.append((ny, nx))
                    else:
                        # 交叉点处理：基于方向一致性和几何特征选择最佳路径
                        if len(current_path) >= 2:
                            # 计算当前方向
                            prev_pos = current_path[-2]
                            curr_pos = current_path[-1]
                            current_dir = _get_direction_vector(prev_pos, curr_pos)
                            
                            # 评估所有候选邻居的综合特征
                            candidates = []
                            for ndy, ndx in nbrs:
                                nny, nnx = ny + ndy, nx + ndx
                                if (0 <= nny < H and 0 <= nnx < W and
                                    skel[nny, nnx] and (nny, nnx) not in current_path):
                                    
                                    # 1. 方向一致性评分
                                    candidate_dir = (ndy, ndx)
                                    angle = _angle_between_vectors(current_dir, candidate_dir)
                                    direction_score = 1.0 - (angle / np.pi)  # 角度越小，分数越高
                                    
                                    # 2. 几何特征分析
                                    curvature, direction_consistency, has_branch = _analyze_line_geometry(
                                        skel, (ny, nx), candidate_dir, length=8
                                    )
                                    
                                    # 综合评分：方向一致性权重0.6，几何特征权重0.4
                                    geometry_score = direction_consistency * 0.7 + (1.0 - min(curvature / np.pi, 1.0)) * 0.3
                                    if has_branch:
                                        geometry_score *= 0.8  # 有分叉的线段降低评分
                                    
                                    total_score = 0.6 * direction_score + 0.4 * geometry_score
                                    
                                    candidates.append((total_score, angle, (nny, nnx), ndy, ndx))
                            
                            # 选择综合评分最高的候选
                            if candidates:
                                candidates.sort(key=lambda x: x[0], reverse=True)  # 按综合评分降序排序
                                best_score, best_angle, best_ny, best_nx, best_dy = candidates[0]
                                
                                # 只有当综合评分较高时才继续
                                if best_score > 0.4:  # 综合评分阈值
                                    dist[ny, nx] = nd
                                    label_map[ny, nx] = label_map[y, x]
                                    path_history[(ny, nx)] = current_path + [(ny, nx)]
                                    q.append((ny, nx))
                        else:
                            # 路径较短，直接分配
                            dist[ny, nx] = nd
                            label_map[ny, nx] = label_map[y, x]
                            path_history[(ny, nx)] = current_path + [(ny, nx)]
                            q.append((ny, nx))
    
    # 应用全局拓扑优化
    label_map = _global_topology_optimization(label_map, skel, intersections)
    
    return label_map


# ---------- 后处理函数 ----------
def _keep_seed_component(labels: np.ndarray, skel: np.ndarray, seed, k: int):
    """只保留与 seed 连通的 k 标签主分量，其余置 0"""
    H, W = labels.shape
    sy, sx = seed
    if not (0 <= sy < H and 0 <= sx < W):
        return labels
    if labels[sy, sx] != k:
        return labels

    keep = np.zeros_like(labels, np.uint8)
    q = [(sy, sx)]
    keep[sy, sx] = 1
    head = 0
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))
    while head < len(q):
        y, x = q[head]; head += 1
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and labels[ny, nx] == k and not keep[ny, nx]:
                keep[ny, nx] = 1
                q.append((ny, nx))

    labels[(labels == k) & (keep == 0)] = 0
    return labels


def _reassign_small_islands(labels: np.ndarray, seeds, max_area: int = 80):
    """将小面积的色块重分配到与其最近的种子标签。"""
    H, W = labels.shape
    out = labels.copy()
    for k in (1, 2, 3):
        mask = (out == k).astype(np.uint8)
        num, comp = cv2.connectedComponents(mask, connectivity=8)
        for cid in range(1, num):
            area = np.sum(comp == cid)
            if area == 0:
                continue
            if area <= max_area:
                ys, xs = np.nonzero(comp == cid)
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                d2 = [(((cy - sy) ** 2 + (cx - sx) ** 2), idx + 1) for idx, (sy, sx) in enumerate(seeds)]
                d2.sort()
                newk = d2[0][1]
                out[(comp == cid)] = newk
    return out


def _prune_short_spurs(skel: np.ndarray, min_len: int = 6):
    """
    在骨架上修剪短枝（度为1的端点往回追，长度<min_len就删除）
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], np.uint8)
    deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.where(deg == 11)

    pruned = skel.copy()
    H, W = skel.shape
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))
    for ey, ex in zip(*endpoints):
        y, x = ey, ex
        path = [(y, x)]
        for _ in range(min_len):
            nxt = None
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and pruned[ny, nx] and (ny, nx) not in path:
                    nxt = (ny, nx)
                    break
            if nxt is None:
                break
            y, x = nxt
            path.append((y, x))
            # 到达分叉（邻域≥4含自身），停止
            if np.sum(pruned[max(0, y - 1): y + 2, max(0, x - 1): x + 2]) >= 4:
                break
        if len(path) <= min_len:
            for (py, px) in path:
                pruned[py, px] = 0
    return pruned


# ---------- 对外主函数 ----------
def colorize_lines(image: np.ndarray | None, fg_mask: np.ndarray | None = None, 
                  seed_points_xy=None, interactive_mode=False, rescue_result=None) -> np.ndarray:
    """
    返回"牵引绳分色"图：三条牵引线用不同颜色标记。
    利用任务2去噪和任务5解救小丑的结果，通过人为选取种子点实现分色。
    
    Args:
        image: 输入图像
        fg_mask: 前景掩膜（可选）
        seed_points_xy: 用户指定的种子点坐标列表 [(x1,y1), (x2,y2), (x3,y3)]
        interactive_mode: 是否启用交互模式，允许用户手动选择种子点
        rescue_result: 任务5的解救小丑结果（可选）
    
    Returns:
        分色后的图像
    """
    # 1. 获取前景掩膜
    if fg_mask is None:
        if rescue_result is not None:
            # 从任务5的结果提取掩膜
            fg = cv2.cvtColor(rescue_result, cv2.COLOR_BGR2GRAY)
            _, fg = cv2.threshold(fg, 1, 255, cv2.THRESH_BINARY)
        else:
            # 使用任务2的去噪结果
            fg = denoising.denoise(image)
    else:
        fg = (fg_mask > 0).astype(np.uint8) * 255
    
    if np.sum(fg > 0) < 500:
        return cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)

    # 2. 生成骨架
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    skel = skeletonize((fg > 0)).astype(np.uint8)
    
    # 3. 交互式种子点选择
    if interactive_mode:
        print("交互模式：请在骨架图像上点击三条牵引线的起点（按F确认，按Q退出）")
        display_img = cv2.cvtColor(skel * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Select Seeds (3 points)", display_img)
        
        # 存储用户点击的点
        clicked_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 在点击位置绘制标记
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Seeds (3 points)", display_img)
                clicked_points.append((x, y))
                print(f"已选择点 {len(clicked_points)}: ({x}, {y})")
        
        cv2.setMouseCallback("Select Seeds (3 points)", mouse_callback)
        
        # 等待用户输入
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f') or key == ord('F'):
                if len(clicked_points) >= 3:
                    print("确认选择，开始处理...")
                    cv2.destroyAllWindows()
                    seed_points_xy = clicked_points[:3]
                    break
                else:
                    print(f"请至少选择3个点，当前已选择 {len(clicked_points)} 个")
            elif key == ord('q') or key == ord('Q'):
                print("取消选择，退出程序")
                cv2.destroyAllWindows()
                return None

    # 4. 使用种子点进行分色
    labels = None
    seeds = None

    # 4.1 使用用户指定的种子点
    if seed_points_xy and len(seed_points_xy) >= 3:
        seeds = []
        for (x, y) in seed_points_xy[:3]:
            pt = _nearest_on_skeleton(int(y), int(x), skel)
            if pt is not None:
                seeds.append(pt)
        if len(seeds) >= 3:
            print(f"使用用户指定的种子点: {seeds}")
            labels = _smart_multisource_bfs_label(skel, seeds)
    
    # 4.2 自动检测种子点（回退策略）
    if labels is None:
        # 尝试底部端点跟踪
        ys, xs = np.nonzero(skel)
        if len(ys) >= 3:
            order = np.argsort(ys)[:3]
            seeds = [(int(xs[i]), int(ys[i])) for i in order]
            labels = _smart_multisource_bfs_label(skel, seeds)
            print(f"使用自动检测的种子点: {seeds}")
        else:
            # 算法失败，返回边缘检测
            dbg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
            edges = cv2.Canny(fg, 50, 150)
            dbg[edges > 0] = (0, 255, 255)
            return dbg

    # 5. 后处理
    for k, sd in zip((1, 2, 3), seeds):
        labels = _keep_seed_component(labels, skel, sd, k)
    labels = _reassign_small_islands(labels, seeds, max_area=80)
    pruned_skel = _prune_short_spurs(skel, min_len=6)
    labels[pruned_skel == 0] = 0

    if np.sum(labels) == 0:
        dbg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
        edges = cv2.Canny(fg, 50, 150)
        dbg[edges > 0] = (0, 255, 255)
        return dbg

    # 6. 着色渲染
    out = np.zeros_like(image, dtype=np.uint8)
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 红、绿、蓝
    present = [k for k in (1, 2, 3) if np.any(labels == k)]
    for color, k in zip(palette, present):
        m = (labels == k).astype(np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)
        out[m > 0] = color

    return out


# ---------- 便捷函数 ----------
def process_pipeline(image, interactive_seed_selection=True):
    """
    完整处理流程：任务2去噪 → 任务5解救小丑 → 任务6牵引绳分色
    
    Args:
        image: 输入图像
        interactive_seed_selection: 是否在任务6中进行交互式种子点选择
    
    Returns:
        分色后的图像
    """
    # 1. 任务2去噪
    denoised = denoising.denoise(image)
    
    # 2. 任务5解救小丑
    rescuer = ClownRescuer()
    rescue_result = rescuer.rescue_clown(image)
    
    # 3. 任务6牵引绳分色
    result = colorize_lines(
        image, 
        rescue_result=rescue_result,
        interactive_mode=interactive_seed_selection
    )
    
    return result
