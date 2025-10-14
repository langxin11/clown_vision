import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from . import denoising  # 复用任务2的掩膜（黑底白前景）


# ---------- 工具：从某点向外找最近的骨架像素 ----------
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


# ---------- 取三枚稳定“种子”：来自三只气球的颈部 ----------
def _get_balloon_seeds(mask: np.ndarray, skel: np.ndarray):
    """
    更稳健的 3 种子选择：
    - 在图像上半区找气球连通域；
    - 取每个连通域 bbox 的下边界中点作为“颈部近似”，在其周围搜最近骨架点；
    - 去重并再次校正到骨架，保证三枚有效且分离的种子。
    """
    H, W = mask.shape

    # 仅在上半区找，避免把小丑身体当成气球
    top = mask.copy()
    top[int(H * 0.55):, :] = 0
    lbl = label(top > 0, connectivity=2)
    regs = [r for r in regionprops(lbl) if r.area > 100]

    if len(regs) < 3:
        # 兜底：整幅再找一次（更靠上的优先）
        lbl = label(mask > 0, connectivity=2)
        regs = [r for r in regionprops(lbl) if r.area > 100]
        regs.sort(key=lambda r: (r.centroid[0], -r.area))

    if not regs:
        return []

    # 取“最上方”的三个
    regs.sort(key=lambda r: r.centroid[0])
    regs = regs[:3]

    raw_seeds = []
    for r in regs:
        minr, minc, maxr, maxc = r.bbox
        cx = (minc + maxc) // 2
        cy = min(maxr + 5, H - 1)  # 向下偏 5 像素，贴近颈部
        pt = _nearest_on_skeleton(cy, cx, skel, max_radius=60)
        if pt is not None:
            raw_seeds.append(pt)

    # 若不足 3 个：从“最上方的骨架点”补齐
    if len(raw_seeds) < 3:
        ys, xs = np.nonzero(skel)
        if len(ys) >= 3:
            order = np.argsort(ys)[:3]
            for i in order:
                p = (int(ys[i]), int(xs[i]))
                if p not in raw_seeds:
                    raw_seeds.append(p)
                    if len(raw_seeds) == 3:
                        break

    # 去重：防止两个种子落得太近
    seeds = []
    MIN_D2 = 25 ** 2  # 最小平方距离
    for py, px in raw_seeds:
        unique = True
        for qy, qx in seeds:
            if (py - qy) ** 2 + (px - qx) ** 2 < MIN_D2:
                unique = False
                break
        if unique:
            seeds.append((py, px))

    # 强制校正到骨架
    fixed = []
    for (py, px) in seeds[:3]:
        if skel[py, px] == 0:
            npy_npx = _nearest_on_skeleton(py, px, skel, max_radius=40)
            if npy_npx is not None:
                fixed.append(npy_npx)
        else:
            fixed.append((py, px))

    return fixed[:3]


# ---------- 在骨架上做三源 BFS（8 邻域） ----------
def _multisource_bfs_label(skel: np.ndarray, seeds):
    """
    在骨架像素上做三源 BFS，返回标签图（0=非骨架；1/2/3=三条线）
    —— 使用 8 邻域，保证斜向骨架连通。
    """
    H, W = skel.shape
    label_map = np.zeros((H, W), np.uint8)
    dist = np.full((H, W), 1_000_000, np.int32)

    q = []
    for i, (y, x) in enumerate(seeds, start=1):
        if 0 <= y < H and 0 <= x < W and skel[y, x]:
            label_map[y, x] = i
            dist[y, x] = 0
            q.append((y, x))

    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))

    head = 0
    while head < len(q):
        y, x = q[head]; head += 1
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                nd = dist[y, x] + 1
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    label_map[ny, nx] = label_map[y, x]
                    q.append((ny, nx))
    return label_map


# ---------- 后处理：仅保留与种子连通的主分量 ----------
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


# ---------- 后处理：小色块重分配到最近种子 ----------
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


# ---------- 后处理：修剪骨架短枝 ----------
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


# ---------- 对外主函数：三色分线 ----------
def colorize_lines(image: np.ndarray) -> np.ndarray:
    """
    返回“牵引绳分色”图：三条牵引线用不同颜色标记。
    若任一步骤失败，回退为在前景边缘上绘制黄色，以避免黑屏。
    """
    # 1) 前景掩膜（白=小丑/气球/线，黑=背景）
    fg = denoising.denoise(image)  # uint8 0/255
    if np.sum(fg > 0) < 500:
        return cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)

    # 2) 轻微闭运算增强连通性 → 骨架
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    skel = skeletonize((fg > 0)).astype(np.uint8)

    # 3) 三个种子（来自三只气球的颈部）
    seeds = _get_balloon_seeds(fg, skel)
    if len(seeds) < 3:
        ys, xs = np.nonzero(skel)
        if len(ys) >= 3:
            order = np.argsort(ys)[:3]
            seeds = [(int(ys[i]), int(xs[i])) for i in order]
        else:
            dbg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
            edges = cv2.Canny(fg, 50, 150)
            dbg[edges > 0] = (0, 255, 255)
            return dbg

    # 4) 多源 BFS 打标签（8 邻域）
    labels = _multisource_bfs_label(skel, seeds)

    # 5) 后处理：主分量 + 小块重分配 + 短枝修剪
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

    # 6) 着色渲染：三色标注，线条稍微加粗以便可视
    out = np.zeros_like(image, dtype=np.uint8)
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 红、绿、蓝
    present = [k for k in (1, 2, 3) if np.any(labels == k)]
    for color, k in zip(palette, present):
        m = (labels == k).astype(np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)
        out[m > 0] = color

    return out

