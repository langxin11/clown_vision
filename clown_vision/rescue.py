import cv2
import numpy as np


def rescue_clown(image):
    # 思路：分割出人物区域，然后取反只保留气球和线
    # (这里只给一个伪代码思路，具体要靠分割/掩膜调试)
    mask_person = np.zeros(image.shape[:2], dtype=np.uint8)
    # TODO: 使用颜色分割/轮廓分割获取人物
    result = cv2.bitwise_and(image, image, mask=~mask_person)
    return result
