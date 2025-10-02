""" 
测试模块rescue.py
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2

# 导入测试模块  
from clown_vision import rescue

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
    
# ==============小丑剔除====================
rescuer = rescue.ClownRescuer()
rescued_img = rescuer.rescue_clown(img)

# =============显示结果===================
if rescued_img is not None:
    #cv2.imwrite("rescued_balloon_with_strings.jpg", rescued_img)
    print("Result saved as 'rescued_balloon_with_strings.jpg'")
    # 显示最终结果
    cv2.imshow("Final Rescued Result (Balloon + Strings)", rescued_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()