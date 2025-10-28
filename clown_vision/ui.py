import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from clown_vision.rescue import ClownRescuer

# 图标目录查找
ICON_DIR = None
current_dir = Path(__file__).parent
for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
    icon_candidate = parent / "icons"
    if icon_candidate.exists():
        ICON_DIR = icon_candidate
        break

from . import denoising, features, preprocessing, untangle
from .config import default_params
from .utils import read_image_any_path, save_image_any_path


# ---------- 工具函数 ----------
def cv2_to_qpixmap(img):
    """Convert OpenCV image to QPixmap."""
    app = QApplication.instance()
    main_window = None
    for widget in app.topLevelWidgets():
        if widget.__class__.__name__ == 'VisionUI':
            main_window = widget
            break

    if main_window:
        max_width = int(main_window.width() * 0.8)
        max_height = int(main_window.height() * 0.8)
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if len(img.shape) == 2:
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------- 后台线程：局部特征 ----------
class LocalFeatureThread(QThread):
    finished = Signal(tuple)
    error = Signal(str)

    def __init__(self, gray, win):
        super().__init__()
        self.gray = gray
        self.win = win

    def run(self):
        try:
            result = features.local_statistics(self.gray, window_size=self.win)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ---------- 主界面类 ----------
class VisionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Clown Project")
        self.resize(1000, 600)
        self.image = None
        self.gray = None
        self.current_result = None
        self.current_image_path = ""
        self.rescued_mask = None  # 由"解救小丑"得到的前景掩膜（仅气球+牵引绳）

        # 状态栏
        self.statusBar = QStatusBar()
        self.statusBar.setFixedHeight(25)
        self.statusLabel = QLabel("就绪")
        self.statusBar.addWidget(self.statusLabel, 1)

        # 界面布局
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        splitter.addWidget(self.create_control_panel())
        splitter.addWidget(self.create_image_panel())
        splitter.setSizes([250, 750])
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.statusBar)
        self.update_status("就绪")

    # ---------- 工具 ----------
    def get_icon(self, name):
        if ICON_DIR:
            path = os.path.join(ICON_DIR, f"{name}.png")
            if os.path.exists(path):
                return QIcon(path)
        return QIcon()

    def update_status(self, msg):
        self.statusLabel.setText(msg)
        self.statusBar.update()

    # ---------- 控制面板 ----------
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 文件操作
        file_group = QGroupBox("文件操作")
        vbox = QVBoxLayout()
        self.btn_load = QPushButton("加载图像")
        self.btn_load.setIcon(self.get_icon("open"))
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setIcon(self.get_icon("save"))
        vbox.addWidget(self.btn_load)
        vbox.addWidget(self.btn_save)
        file_group.setLayout(vbox)

        # 预处理
        pre_group = QGroupBox("图像预处理")
        grid = QGridLayout()
        self.btn_gray = QPushButton("灰度化")
        self.btn_binary = QPushButton("二值化")
        self.btn_fourier = QPushButton("傅里叶")
        self.btn_denoise = QPushButton("去噪")
        for btn in [self.btn_gray, self.btn_binary, self.btn_fourier, self.btn_denoise]:
            btn.setFixedHeight(35)
        grid.addWidget(self.btn_gray, 0, 0)
        grid.addWidget(self.btn_binary, 0, 1)
        grid.addWidget(self.btn_fourier, 1, 0)
        grid.addWidget(self.btn_denoise, 1, 1)
        pre_group.setLayout(grid)

        # 特征提取
        feat_group = QGroupBox("特征提取")
        grid2 = QGridLayout()
        self.btn_local = QPushButton("局部特征")
        self.btn_lbp = QPushButton("LBP")
        self.btn_hog = QPushButton("HOG")
        self.btn_haar = QPushButton("Haar(示例)")
        for btn in [self.btn_local, self.btn_lbp, self.btn_hog, self.btn_haar]:
            btn.setFixedHeight(35)
        grid2.addWidget(self.btn_local, 0, 0)
        grid2.addWidget(self.btn_lbp, 0, 1)
        grid2.addWidget(self.btn_hog, 1, 0)
        grid2.addWidget(self.btn_haar, 1, 1)
        feat_group.setLayout(grid2)

        # 特殊功能
        sp_group = QGroupBox("特殊功能")
        vsp = QVBoxLayout()
        self.btn_rescue = QPushButton("解救小丑")
        self.btn_untangle = QPushButton("牵引绳分割")
        self.btn_untangle_seed = QPushButton("牵引绳分割(手动选种子)")
        for btn in [self.btn_rescue, self.btn_untangle, self.btn_untangle_seed]:
            btn.setFixedHeight(35)
        vsp.addWidget(self.btn_rescue)
        vsp.addWidget(self.btn_untangle)
        vsp.addWidget(self.btn_untangle_seed)
        sp_group.setLayout(vsp)

        # 参数设置
        param_group = QGroupBox("参数设置")
        vparam = QVBoxLayout()
        self.spin_win = QSpinBox()
        self.spin_win.setRange(3, 101)
        self.spin_win.setSingleStep(2)
        self.spin_win.setValue(default_params.sliding_window_size)
        self.spin_win.setPrefix("滑窗:")
        vparam.addWidget(self.spin_win)
        label = QLabel("滑窗：用于局部特征计算的窗口大小")
        label.setWordWrap(True)
        vparam.addWidget(label)
        param_group.setLayout(vparam)

        # 布局组合
        for g in [file_group, pre_group, feat_group, sp_group, param_group]:
            layout.addWidget(g)
        layout.addStretch()

        # 信号连接
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_gray.clicked.connect(self.show_gray)
        self.btn_binary.clicked.connect(self.show_binary)
        self.btn_fourier.clicked.connect(self.show_fourier)
        self.btn_denoise.clicked.connect(self.show_denoise)
        self.btn_local.clicked.connect(self.show_local)
        self.btn_lbp.clicked.connect(self.show_lbp)
        self.btn_hog.clicked.connect(self.show_hog)
        self.btn_haar.clicked.connect(self.show_haar)
        self.btn_rescue.clicked.connect(self.show_rescue)
        self.btn_untangle.clicked.connect(self.show_untangle)
        self.btn_untangle_seed.clicked.connect(self.show_untangle_with_seeds)
        self.spin_win.valueChanged.connect(self.update_params)
        return panel

    # ---------- 图像显示面板 ----------
    def create_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.label = QLabel("No Image")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        scroll.setWidget(self.label)
        layout.addWidget(scroll)
        return panel

    # ---------- 各类按钮功能 ----------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            self.update_status("已取消图像选择")
            return
        try:
            self.image = read_image_any_path(path)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.current_result = self.image
            self.label.setPixmap(cv2_to_qpixmap(self.image))
            h, w = self.image.shape[:2]
            self.update_status(f"已加载图像: {os.path.basename(path)} ({w}x{h})")
        except Exception as e:
            self.label.setText(f"加载失败: {e}")
            self.update_status("加载图像出错")

    def show_gray(self):
        if self.gray is None:
            self.update_status("请先加载图像")
            return
        self.current_result = self.gray
        self.label.setPixmap(cv2_to_qpixmap(self.gray))
        self.update_status("已显示灰度图像")

    def show_binary(self):
        if self.gray is None:
            self.update_status("请先加载图像")
            return
        binary = preprocessing.to_binary(self.gray)
        self.current_result = binary
        self.label.setPixmap(cv2_to_qpixmap(binary))
        self.update_status("已显示二值化图像")

    def show_fourier(self):
        if self.gray is None:
            self.update_status("请先加载图像")
            return
        spectrum = preprocessing.fourier_transform(self.gray)
        self.current_result = spectrum
        self.label.setPixmap(cv2_to_qpixmap(spectrum))
        self.update_status("已显示傅里叶频谱")

    def show_denoise(self):
        if self.image is None:
            self.update_status("请先加载图像")
            return
        denoised = denoising.denoise(self.image)
        self.current_result = denoised
        self.label.setPixmap(cv2_to_qpixmap(denoised))
        self.update_status("已显示去噪结果")

    # ---------- 局部特征（异步） ----------
    def show_local(self):
        if self.gray is None:
            self.update_status("请先加载图像")
            return
        win = int(max(3, self.spin_win.value() // 2 * 2 + 1))
        self.update_status(f"正在计算局部特征 (窗口大小: {win})...")
        self.label.setText("⏳ 正在计算，请稍候...")
        self.thread = LocalFeatureThread(self.gray, win)
        self.thread.finished.connect(self.on_local_finished)
        self.thread.error.connect(self.on_local_error)
        self.thread.start()

    def on_local_finished(self, result):
        # 期望收到四幅 0-255 灰度图：mean / moment1 / moment2 / entropy
        local_mean, moment1, moment2, entropy_img = result
        combined = np.hstack([local_mean, moment1, moment2, entropy_img])
        self.current_result = combined
        self.label.setPixmap(cv2_to_qpixmap(combined))
        self.update_status("局部特征：均值 / 一阶中心绝对矩 / 二阶中心矩 / 熵")

    def on_local_error(self, msg):
        self.label.setText(f"❌ 错误: {msg}")
        self.update_status(f"局部特征计算异常: {msg}")

    # ---------- 其他特征 ----------
    def show_lbp(self):
        if self.gray is None:
            self.update_status("请先加载图像")
            return
        lbp = features.extract_lbp(self.gray)
        self.current_result = lbp
        self.label.setPixmap(cv2_to_qpixmap(lbp))
        self.update_status("已显示 LBP 特征")

    def show_hog(self):
        if self.gray is None:
            self.update_status("请先加载图像")
            return
        hog_img = features.extract_hog(self.gray)
        self.current_result = hog_img
        self.label.setPixmap(cv2_to_qpixmap(hog_img))
        self.update_status("已显示 HOG 特征")

    def show_haar(self):
        if self.image is None:
            self.update_status("请先加载图像")
            return
        haar_img = features.extract_haar(self.image)
        self.current_result = haar_img
        self.label.setPixmap(cv2_to_qpixmap(haar_img))
        self.update_status("已显示 Haar 检测结果")

    # ---------- 特殊功能 ----------
    def show_rescue(self):
        if self.image is None:
            self.update_status("请先加载图像")
            return
        QMessageBox.information(self, "说明", "点击并拖动选择要保留的区域，按F确认，Q退出。")
        rescuer = ClownRescuer()
        result = rescuer.rescue_clown(self.image)
        if result is not None:
            self.current_result = result
            self.label.setPixmap(cv2_to_qpixmap(result))
            self.update_status("已完成小丑解救")
            # 直接使用救援流程产生的"非小丑前景掩膜"（仅气球+牵引绳）
            self.rescued_mask = getattr(rescuer, 'last_nonclown_foreground_mask', None)

    def show_untangle(self):
        if self.image is None:
            self.update_status("请先加载图像")
            return
        # 若已有"解救小丑"的前景掩膜，则复用以避免小丑边缘干扰
        result = untangle.colorize_lines(self.image, getattr(self, 'rescued_mask', None))
        self.current_result = result
        self.label.setPixmap(cv2_to_qpixmap(result))
        self.update_status("已显示牵引绳分割结果")

    def show_untangle_with_seeds(self):
        if self.image is None:
            self.update_status("请先加载图像")
            return
        QMessageBox.information(self, "说明", "请在骨架图像上依次点击三条牵引绳的起点，然后按F确认，Q取消")
        
        # 获取任务5的结果（如果已执行）
        rescue_result = None
        if hasattr(self, 'current_result') and self.current_result is not None:
            # 检查当前结果是否是解救小丑的结果
            rescue_result = self.current_result
        
        # 使用改进的牵引绳分色算法，在骨架图像上选择种子点
        result = untangle.colorize_lines(
            self.image, 
            fg_mask=getattr(self, 'rescued_mask', None),
            rescue_result=rescue_result,
            interactive_mode=True
        )
        
        if result is not None:
            self.current_result = result
            self.label.setPixmap(cv2_to_qpixmap(result))
            self.update_status("已显示牵引绳分割结果（手动选种子）")
        else:
            self.update_status("已取消种子选择")

    # ---------- 保存 ----------
    def save_current(self):
        if self.current_result is None:
            self.update_status("无结果可保存")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.png", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        ok = save_image_any_path(path, self.current_result)
        if ok:
            self.update_status(f"已保存到: {os.path.basename(path)}")
        else:
            self.update_status("保存失败")

    def update_params(self, val):
        win = int(max(3, val // 2 * 2 + 1))
        default_params.sliding_window_size = win
        self.update_status(f"滑窗大小更新为 {win}")

# ---------- 主入口 ----------
def main():
    app = QApplication(sys.argv)
    w = VisionUI()
    w.show()
    sys.exit(app.exec())
