import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

# 导入测试模块  
from clown_vision import preprocessing  #包含灰度化函数、二值化、Fourier变换函数

# ==============加载图片====================
img = cv2.imread("assets/test.png")
# 检查图片是否成功加载
if img is None:
    print("错误：无法加载图像，请检查文件路径")
else:
    # 显示图片
    cv2.imshow("Image", img)
    # 等待按键，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================测试灰度化函数====================
gray = preprocessing.to_gray(img)
# 检查灰度化结果是否正确
assert gray.shape == img.shape[:2], "灰度化后图像形状与原图像不同"
# 显示灰度化结果
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#后续添加测试代码
# ================测试二值化函数====================
binary = preprocessing.to_binary(gray, use_otsu=True)
binary = preprocessing.to_binary(gray, thresh=240,use_otsu=False)
cv2.imshow("Binary Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===========测试fuourier_transform函数================
fourier = preprocessing.fourier_transform(gray)
cv2.imshow("Fourier Transform", fourier)
cv2.waitKey(0)
cv2.destroyAllWindows()

#==============保存图片====================
cv2.imwrite("assets/gray.png", gray)
cv2.imwrite("assets/binary.png", binary)
cv2.imwrite("assets/fourier.png", fourier)
