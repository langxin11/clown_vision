"""
测试denoise.py模块
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2

from clown_vision import denoising

# ==============加载图片====================
img = cv2.imread("assets/test.png")
if img is None:
    print("错误：无法加载图像，请检查文件路径")
de_img = denoising.denoise(img)
#保存图像
denoised_path = "assets/denoised.png"
cv2.imwrite(denoised_path, de_img)
cv2.imshow("denoised", de_img)
cv2.waitKey(0)
cv2.destroyAllWindows()