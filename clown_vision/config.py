from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcessingParams:
    """Centralized parameters for image processing steps.

    Extend as needed; UI can bind to these values.
    """

    # preprocessing
    binary_threshold: int = 128

    # local features
    sliding_window_size: int = 15

    # denoising
    denoise_blur_ksize: int = 5
    denoise_blur_sigma: float = 1.5
    denoise_adaptive: bool = False
    denoise_adaptive_block: int = 21
    denoise_adaptive_C: int = 7
    denoise_fixed_thresh: int = 230
    denoise_invert: bool = True  # 与原实现一致：前景白使用 INV 阈值
    morph_kernel: int = 3
    morph_open_iters: int = 1
    morph_close_iters: int = 2

    # rescue (clown removal)
    rescue_dilate_clown: int = 2  # 膨胀小丑掩膜的迭代次数，消除边缘残留


# A single shared default instance for simple use-cases
default_params = ProcessingParams()


