import sys

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import denoising, descriptors, local_features, preprocessing
from .config import default_params
from .utils import read_image_any_path, save_image_any_path


def cv2_to_qpixmap(img):
    """Convert OpenCV image to QPixmap."""
    # 获取当前窗口大小
    app = QApplication.instance()
    main_window = None
    for widget in app.topLevelWidgets():
        # 使用类名检查，避免循环导入问题
        if widget.__class__.__name__ == 'VisionUI':
            main_window = widget
            break
    
    if main_window:
        # 计算最大允许的图像尺寸（窗口的80%）
        max_width = int(main_window.width() * 0.8)
        max_height = int(main_window.height() * 0.8)
        
        # 获取原始图像尺寸
        h, w = img.shape[:2]
        
        # 计算缩放比例，确保图像不会超过最大允许尺寸
        scale = min(max_width / w, max_height / h)
        
        # 如果图像需要缩小
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            # 使用cv2.resize调整图像大小
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 正常的OpenCV到QPixmap转换
    if len(img.shape) == 2:  # gray
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    else:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    return QPixmap.fromImage(qimg)

class VisionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Clown Project")
        # 设置窗口默认大小为 800x600 像素
        self.resize(800, 600)
        self.image = None  # 原始图像
        self.gray = None   # 灰度图
        self.current_result = None  # 最近一次结果（用于保存）

        # UI 布局
        layout = QHBoxLayout(self)

        # 左边按钮
        btn_layout = QVBoxLayout()
        self.btn_load = QPushButton("加载图像")
        self.btn_gray = QPushButton("灰度化")
        self.btn_binary = QPushButton("二值化")
        self.btn_fourier = QPushButton("傅里叶")
        self.btn_denoise = QPushButton("去噪")
        self.btn_local = QPushButton("局部特征")
        self.btn_lbp = QPushButton("LBP")
        self.btn_hog = QPushButton("HOG")
        self.btn_haar = QPushButton("Haar")
        # 参数：滑窗大小
        self.spin_win = QSpinBox()
        self.spin_win.setRange(3, 101)
        self.spin_win.setSingleStep(2)
        self.spin_win.setValue(default_params.sliding_window_size)
        self.spin_win.setPrefix("滑窗:")
        # 保存按钮
        self.btn_save = QPushButton("保存结果")

        for b in [
            self.btn_load, self.btn_gray, self.btn_binary, self.btn_fourier,
            self.btn_denoise, self.btn_local, self.btn_lbp, self.btn_hog, self.btn_haar,
            self.spin_win, self.btn_save,
        ]:
            btn_layout.addWidget(b)

        layout.addLayout(btn_layout)

        # 右边显示图像
        self.label = QLabel("No Image")
        self.label.setScaledContents(True)
        layout.addWidget(self.label, 1)

        # 信号连接
        self.btn_load.clicked.connect(self.load_image)
        self.btn_gray.clicked.connect(self.show_gray)
        self.btn_binary.clicked.connect(self.show_binary)
        self.btn_fourier.clicked.connect(self.show_fourier)
        self.btn_denoise.clicked.connect(self.show_denoise)
        self.btn_local.clicked.connect(self.show_local)
        self.btn_lbp.clicked.connect(self.show_lbp)
        self.btn_hog.clicked.connect(self.show_hog)
        self.btn_haar.clicked.connect(self.show_haar)
        self.btn_save.clicked.connect(self.save_current)
        self.spin_win.valueChanged.connect(self.update_params)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.bmp)")
        if path:
            # 使用工具函数读取以支持中文路径
            self.image = read_image_any_path(path)
            if self.image is not None:
                self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.current_result = self.image
                # 窗口保持固定大小，只更新图像内容
                self.label.setPixmap(cv2_to_qpixmap(self.image))
                # 确保label的缩放属性是启用的，这样图片会自动适应label大小
                self.label.setScaledContents(True)
            else:
                self.label.setText("无法加载图像，请检查文件路径或文件格式")
                self.image = None
                self.gray = None

    def show_gray(self):
        if self.gray is not None:
            self.current_result = self.gray
            self.label.setPixmap(cv2_to_qpixmap(self.gray))

    def show_binary(self):
        if self.gray is not None:
            binary = preprocessing.to_binary(self.gray)
            self.current_result = binary
            self.label.setPixmap(cv2_to_qpixmap(binary))

    def show_fourier(self):
        if self.gray is not None:
            spectrum = preprocessing.fourier_transform(self.gray)
            self.current_result = spectrum
            self.label.setPixmap(cv2_to_qpixmap(spectrum))

    def show_denoise(self):
        if self.image is not None:
            denoised = denoising.denoise(self.image)
            self.current_result = denoised
            self.label.setPixmap(cv2_to_qpixmap(denoised))

    def show_local(self):
        if self.gray is not None:
            win = int(max(3, self.spin_win.value() // 2 * 2 + 1))
            default_params.sliding_window_size = win
            feat = local_features.sliding_window_features(self.gray, win_size=win)
            self.current_result = feat
            self.label.setPixmap(cv2_to_qpixmap(feat))

    def show_lbp(self):
        if self.gray is not None:
            lbp = descriptors.compute_lbp(self.gray)
            self.current_result = lbp
            self.label.setPixmap(cv2_to_qpixmap(lbp))

    def show_hog(self):
        if self.gray is not None:
            hog = descriptors.compute_hog(self.gray)
            self.current_result = hog
            self.label.setPixmap(cv2_to_qpixmap(hog))

    def show_haar(self):
        if self.gray is not None:
            haar = descriptors.compute_haar(self.gray)
            # Haar 是数值特征，不是图像；这里先展示为灰度条
            import numpy as np
            h_img = np.zeros((100, len(haar)), dtype=np.uint8)
            haar_norm = np.interp(haar, (haar.min(), haar.max()), (0, 255)).astype(np.uint8)
            h_img[:] = haar_norm
            self.current_result = h_img
            self.label.setPixmap(cv2_to_qpixmap(h_img))

    def update_params(self, value: int):
        # 统一奇数窗口大小
        win = int(max(3, value // 2 * 2 + 1))
        if win != value:
            self.spin_win.blockSignals(True)
            self.spin_win.setValue(win)
            self.spin_win.blockSignals(False)
        default_params.sliding_window_size = win

    def save_current(self):
        if self.current_result is None:
            self.label.setText("无可保存结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.png", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        ok = save_image_any_path(path, self.current_result)
        if not ok:
            self.label.setText("保存失败，请检查路径或格式")

def main():
    app = QApplication(sys.argv)
    w = VisionUI()
    w.show()
    sys.exit(app.exec())
