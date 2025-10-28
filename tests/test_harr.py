import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2


def extract_haar(image):
    """提取 Haar 特征（例如检测小丑或气球等目标）

    使用 OpenCV 的 Haar 级联分类器（示例：人脸检测）

    Args:
        image (numpy.ndarray): 输入彩色图像

    Returns:
        numpy.ndarray: 绘制检测框后的输出图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化增强对比度
    gray = cv2.equalizeHist(gray)

     # 或者使用CLAHE（限制对比度自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    detections = classifier.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10),flags=cv2.CASCADE_SCALE_IMAGE)
    output = image.copy()

    for (x, y, w, h) in detections:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return output

### 测试harr特征提取
img = cv2.imread("assets/test.png")
out = extract_haar(img)
cv2.imshow("Image", out)
cv2.waitKey(0)
cv2.destroyAllWindows()