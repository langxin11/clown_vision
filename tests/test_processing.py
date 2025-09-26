import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

# 导入测试模块
import clown_vision.preprocessing as preprocessing

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

# 测试灰度化函数
gray = preprocessing.to_gray(img)
# 检查灰度化结果是否正确
assert gray.shape == img.shape[:2], "灰度化后图像形状与原图像不同"
# 显示灰度化结果
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#后续添加测试代码
