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
    bilateral_diameter: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75


# A single shared default instance for simple use-cases
default_params = ProcessingParams()


