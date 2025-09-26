import cv2
import numpy as np

#from skimage.util.shape import view_as_windows
from skimage.measure import shannon_entropy


def sliding_window_features(gray, win_size=15):
    h, w = gray.shape
    pad = win_size // 2
    padded = np.pad(gray, pad, mode="reflect")
    out = np.zeros_like(gray, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+win_size, j:j+win_size]
            # 局部均值/矩/熵
            mean = patch.mean()
            moment1 = np.mean((patch - mean))
            moment2 = np.mean((patch - mean)**2)
            entropy = shannon_entropy(patch)

            out[i,j] = mean + moment1 + moment2 + entropy

    out_norm = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out_norm.astype(np.uint8)
