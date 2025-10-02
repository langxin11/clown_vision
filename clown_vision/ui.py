""" 
用户界面
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from clown_vision.rescue import ClownRescuer

# 为了图标支持，尝试查找icons目录
ICON_DIR = None
current_dir = Path(__file__).parent
for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
    icon_candidate = parent / "icons"
    if icon_candidate.exists():
        ICON_DIR = icon_candidate
        break

from . import denoising, descriptors, local_features, preprocessing, untangle
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
        # 设置窗口默认大小为 1000x600 像素
        self.resize(1000, 600)
        self.image = None  # 原始图像
        self.gray = None   # 灰度图
        self.current_result = None  # 最近一次结果（用于保存）
        self.current_image_path = ""  # 当前图像路径
        
        # 初始化状态栏
        self.statusBar = QStatusBar()
        self.statusBar.setFixedHeight(25)
        self.statusLabel = QLabel("就绪")
        self.statusBar.addWidget(self.statusLabel, 1)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建分隔器，让UI更灵活
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧图像显示区域
        image_panel = self.create_image_panel()
        splitter.addWidget(image_panel)
        
        # 设置初始分割比例
        splitter.setSizes([250, 750])  # 左侧25%，右侧75%
        
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.statusBar)
        
        # 更新状态栏
        self.update_status("就绪")
        
    def create_menu_bar(self):
        """创建菜单栏"""
        # 为了简化实现，这里暂时不添加完整的菜单栏
        pass
    
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        panel_layout.setSpacing(5)
        
        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.btn_load = QPushButton("加载图像")
        self.btn_load.setIcon(self.get_icon("open"))
        self.btn_load.setFixedHeight(40)
        
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setIcon(self.get_icon("save"))
        self.btn_save.setFixedHeight(40)
        
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.btn_save)
        file_group.setLayout(file_layout)
        
        # 图像预处理组
        preprocess_group = QGroupBox("图像预处理")
        preprocess_layout = QVBoxLayout()
        
        # 创建按钮网格布局
        preprocess_grid = QGridLayout()
        preprocess_grid.setSpacing(5)
        
        self.btn_gray = QPushButton("灰度化")
        self.btn_gray.setFixedHeight(35)
        self.btn_binary = QPushButton("二值化")
        self.btn_binary.setFixedHeight(35)
        self.btn_fourier = QPushButton("傅里叶")
        self.btn_fourier.setFixedHeight(35)
        self.btn_denoise = QPushButton("去噪")
        self.btn_denoise.setFixedHeight(35)
        
        preprocess_grid.addWidget(self.btn_gray, 0, 0)
        preprocess_grid.addWidget(self.btn_binary, 0, 1)
        preprocess_grid.addWidget(self.btn_fourier, 1, 0)
        preprocess_grid.addWidget(self.btn_denoise, 1, 1)
        
        preprocess_layout.addLayout(preprocess_grid)
        preprocess_group.setLayout(preprocess_layout)
        
        # 特征提取组
        feature_group = QGroupBox("特征提取")
        feature_layout = QVBoxLayout()
        
        feature_grid = QGridLayout()
        feature_grid.setSpacing(5)
        
        self.btn_local = QPushButton("局部特征")
        self.btn_local.setFixedHeight(35)
        self.btn_lbp = QPushButton("LBP")
        self.btn_lbp.setFixedHeight(35)
        self.btn_hog = QPushButton("HOG")
        self.btn_hog.setFixedHeight(35)
        self.btn_haar = QPushButton("Haar")
        self.btn_haar.setFixedHeight(35)
        
        feature_grid.addWidget(self.btn_local, 0, 0)
        feature_grid.addWidget(self.btn_lbp, 0, 1)
        feature_grid.addWidget(self.btn_hog, 1, 0)
        feature_grid.addWidget(self.btn_haar, 1, 1)
        
        feature_layout.addLayout(feature_grid)
        feature_group.setLayout(feature_layout)
        
        # 特殊功能组
        special_group = QGroupBox("特殊功能")
        special_layout = QVBoxLayout()
        
        self.btn_rescue = QPushButton("解救小丑")
        self.btn_rescue.setFixedHeight(35)
        
        self.btn_untangle = QPushButton("牵引绳分割")
        self.btn_untangle.setFixedHeight(35)
        
        special_layout.addWidget(self.btn_rescue)
        special_layout.addWidget(self.btn_untangle)
        special_group.setLayout(special_layout)
        
        # 参数设置组
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout()
        
        # 滑窗大小参数
        self.spin_win = QSpinBox()
        self.spin_win.setRange(3, 101)
        self.spin_win.setSingleStep(2)
        self.spin_win.setValue(default_params.sliding_window_size)
        self.spin_win.setPrefix("滑窗:")
        
        # 添加滑窗参数解释
        window_explanation = QLabel("滑窗：用于局部特征计算的窗口大小\n")
        window_explanation.setWordWrap(True)
        window_explanation.setAlignment(Qt.AlignTop)
        window_explanation.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        param_layout.addWidget(self.spin_win)
        param_layout.addWidget(window_explanation)
        param_group.setLayout(param_layout)
        
        # 添加所有组到面板布局
        panel_layout.addWidget(file_group)
        panel_layout.addWidget(preprocess_group)
        panel_layout.addWidget(feature_group)
        panel_layout.addWidget(special_group)
        panel_layout.addWidget(param_group)
        panel_layout.addStretch()  # 确保下方有空白
        
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
        self.btn_untangle.clicked.connect(self.show_untangle)
        self.btn_rescue.clicked.connect(self.show_rescue)
        
        return panel
    
    def create_image_panel(self):
        """创建右侧图像显示区域"""
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        panel_layout.setSpacing(5)
        
        # 图像显示区域（带滚动条）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.label = QLabel("No Image")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(400, 300)
        
        scroll_area.setWidget(self.label)
        
        panel_layout.addWidget(scroll_area)
        
        return panel
    
    def get_icon(self, name):
        """尝试获取图标"""
        if ICON_DIR is not None:
            icon_path = os.path.join(ICON_DIR, f"{name}.png")
            if os.path.exists(icon_path):
                return QIcon(icon_path)
        # 如果图标不存在，返回空图标
        return QIcon()
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.statusLabel.setText(message)
        self.statusBar.update()

    def load_image(self):
        """加载图像文件"""
        self.update_status("选择图像文件...")
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.update_status(f"正在加载图像: {os.path.basename(path)}...")
            # 使用工具函数读取以支持中文路径
            try:
                self.image = read_image_any_path(path)
                if self.image is not None:
                    self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    self.current_result = self.image
                    self.current_image_path = path
                    
                    # 窗口保持固定大小，只更新图像内容
                    self.label.setPixmap(cv2_to_qpixmap(self.image))
                    # 确保label的缩放属性是启用的，这样图片会自动适应label大小
                    self.label.setScaledContents(True)
                    
                    h, w = self.image.shape[:2]
                    self.update_status(f"已加载图像: {os.path.basename(path)} ({w}x{h})")
                else:
                    self.label.setText("无法加载图像，请检查文件路径或文件格式")
                    self.image = None
                    self.gray = None
                    self.current_image_path = ""
                    self.update_status("加载图像失败")
            except Exception as e:
                self.label.setText(f"加载图像出错: {str(e)}")
                self.image = None
                self.gray = None
                self.current_image_path = ""
                self.update_status(f"加载图像异常: {str(e)}")
        else:
            self.update_status("已取消图像选择")

    def show_gray(self):
        """显示灰度图像"""
        if self.gray is not None:
            self.update_status("正在转换为灰度图像...")
            try:
                self.current_result = self.gray
                self.label.setPixmap(cv2_to_qpixmap(self.gray))
                self.update_status("已显示灰度图像")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"灰度化处理异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_binary(self):
        """显示二值化图像"""
        if self.gray is not None:
            self.update_status("正在进行二值化处理...")
            try:
                binary = preprocessing.to_binary(self.gray)
                self.current_result = binary
                self.label.setPixmap(cv2_to_qpixmap(binary))
                self.update_status("已显示二值化图像")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"二值化处理异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_fourier(self):
        """显示傅里叶变换结果"""
        if self.gray is not None:
            self.update_status("正在进行傅里叶变换...")
            try:
                spectrum = preprocessing.fourier_transform(self.gray)
                self.current_result = spectrum
                self.label.setPixmap(cv2_to_qpixmap(spectrum))
                self.update_status("已显示傅里叶变换频谱")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"傅里叶变换异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_denoise(self):
        """显示去噪图像"""
        if self.image is not None:
            self.update_status("正在进行图像去噪...")
            try:
                denoised = denoising.denoise(self.image)
                self.current_result = denoised
                self.label.setPixmap(cv2_to_qpixmap(denoised))
                self.update_status("已显示去噪后的图像")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"图像去噪异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_local(self):
        """显示局部特征"""
        if self.gray is not None:
            win = int(max(3, self.spin_win.value() // 2 * 2 + 1))
            default_params.sliding_window_size = win
            self.update_status(f"正在提取局部特征 (窗口大小: {win})...")
            try:
                feat = local_features.sliding_window_features(self.gray, win_size=win)
                self.current_result = feat
                self.label.setPixmap(cv2_to_qpixmap(feat))
                self.update_status(f"已显示局部特征 (窗口大小: {win})")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"局部特征提取异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_lbp(self):
        """显示LBP特征"""
        if self.gray is not None:
            self.update_status("正在计算LBP特征...")
            try:
                lbp = descriptors.compute_lbp(self.gray)
                self.current_result = lbp
                self.label.setPixmap(cv2_to_qpixmap(lbp))
                self.update_status("已显示LBP特征图像")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"LBP特征计算异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_hog(self):
        """显示HOG特征"""
        if self.gray is not None:
            self.update_status("正在计算HOG特征...")
            try:
                hog = descriptors.compute_hog(self.gray)
                self.current_result = hog
                self.label.setPixmap(cv2_to_qpixmap(hog))
                self.update_status("已显示HOG特征图像")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"HOG特征计算异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_haar(self):
        """显示Haar特征"""
        if self.gray is not None:
            self.update_status("正在计算Haar特征...")
            try:
                haar = descriptors.compute_haar(self.gray)
                # Haar 是数值特征，不是图像；这里先展示为灰度条
                h_img = np.zeros((100, len(haar)), dtype=np.uint8)
                haar_norm = np.interp(haar, (haar.min(), haar.max()), (0, 255)).astype(np.uint8)
                h_img[:] = haar_norm
                self.current_result = h_img
                self.label.setPixmap(cv2_to_qpixmap(h_img))
                self.update_status("已显示Haar特征可视化")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"Haar特征计算异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def update_params(self, value: int):
        """更新参数设置"""
        # 统一奇数窗口大小
        win = int(max(3, value // 2 * 2 + 1))
        if win != value:
            self.spin_win.blockSignals(True)
            self.spin_win.setValue(win)
            self.spin_win.blockSignals(False)
        default_params.sliding_window_size = win
        self.update_status(f"已更新窗口大小: {win}")

    def save_current(self):
        """保存当前结果图像"""
        if self.current_result is None:
            self.label.setText("无可保存结果")
            self.update_status("无可保存结果")
            return
        
        self.update_status("选择保存路径...")
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.png", "Images (*.png *.jpg *.bmp)")
        
        if path:
            self.update_status(f"正在保存结果到: {os.path.basename(path)}...")
            try:
                ok = save_image_any_path(path, self.current_result)
                if ok:
                    self.update_status(f"已成功保存结果到: {os.path.basename(path)}")
                else:
                    self.label.setText("保存失败，请检查路径或格式")
                    self.update_status("保存结果失败")
            except Exception as e:
                self.label.setText(f"保存图像出错: {str(e)}")
                self.update_status(f"保存结果异常: {str(e)}")
        else:
            self.update_status("已取消保存操作")

    def show_rescue(self):
        """解救小丑功能：交互式剔除小丑区域，保留气球和牵引线"""
        if self.image is not None:
            # 显示操作提示
            QMessageBox.information(
                self,
                "操作提示",
                "解救小丑功能说明：\n1. 系统会自动检测鲜艳色区域作为初始框选\n2. 鼠标框选需要保留的区域，松开鼠标会自动执行GrabCut分割\n3. 多次框选可以添加更多需要保留的区域\n4. 按F键确认最终结果，按Q键退出操作"
            )
            
            self.update_status("正在解救小丑...")
            try:
                rescuer = ClownRescuer()
                result = rescuer.rescue_clown(self.image)
                if result is not None:
                    self.current_result = result
                    self.label.setPixmap(cv2_to_qpixmap(result))
                    self.update_status("已完成小丑解救")
                else:
                    self.update_status("已取消小丑解救操作")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"解救小丑异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

    def show_untangle(self):
        """显示牵引绳分割结果"""
        if self.image is not None:
            self.update_status("正在进行牵引绳分割...")
            try:
                result = untangle.colorize_lines(self.image)
                self.current_result = result
                self.label.setPixmap(cv2_to_qpixmap(result))
                self.update_status("已显示牵引绳分割结果")
            except Exception as e:
                self.label.setText(f"处理图像出错: {str(e)}")
                self.update_status(f"牵引绳分割异常: {str(e)}")
        else:
            self.update_status("请先加载图像")

def main():
    app = QApplication(sys.argv)
    w = VisionUI()
    w.show()
    sys.exit(app.exec())
