""" 
小丑人物剔除模块（面向对象版）
功能：自动检测小丑区域 + 鼠标交互补选遗漏区域 + 剔除小丑，保留气球与牵引线
交互逻辑：
1. 程序启动自动显示初始检测框
2. 鼠标拖动框选遗漏区域，松开鼠标自动执行GrabCut并合并掩码
3. 所有区域补选完成后，按F键确认最终结果，按Q键退出
"""
import cv2
import numpy as np


class ClownRescuer:
    def __init__(self):
        # 类属性（替代全局变量，封装交互状态与参数）
        self.rect = None        # 当前鼠标框选的矩形（x, y, w, h）
        self.drawing = False    # 是否正在拖动鼠标框选
        self.start_x = 0        # 鼠标框选起点x坐标
        self.start_y = 0        # 鼠标框选起点y坐标
        self.img_copy = None    # 图像副本（用于实时绘制框选）
        self.masks = []         # 存储多次分割的掩码（用于合并）
        self.current_img = None # 当前处理的原始图像
        self.result_win_name = "Current Result (Press F to confirm, Q to quit)"  # 结果显示窗口名
        self.select_win_name = "Select Clown (Drag to add area)"  # 框选窗口名

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数（内部方法，处理框选逻辑）"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标按下：记录起点，准备框选
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.img_copy = self.current_img.copy()  # 保存当前图像副本，避免绘制残留

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 鼠标拖动：实时绘制矩形框
            temp_img = self.img_copy.copy()
            cv2.rectangle(temp_img, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow(self.select_win_name, temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            # 鼠标松开：计算框选矩形，执行GrabCut并合并掩码
            self.drawing = False
            end_x, end_y = x, y
            # 确保矩形宽高为正（处理鼠标从右下往左上拖动的情况）
            rect_x = min(self.start_x, end_x)
            rect_y = min(self.start_y, end_y)
            rect_w = abs(end_x - self.start_x)
            rect_h = abs(end_y - self.start_y)

            # 过滤过小的框选（避免误操作）
            if rect_w > 20 and rect_h > 20:
                self.rect = (rect_x, rect_y, rect_w, rect_h)
                print(f"Added area: {self.rect} (auto-run GrabCut)")
                # 对当前框选区域执行GrabCut
                new_mask = self._grabcut_segment(self.current_img, self.rect)
                self.masks.append(new_mask)
                # 合并所有掩码并更新显示结果
                self._update_result_display()
            else:
                print("Skipped small area (width/height < 20)")
            # 重置框选状态
            self.rect = None

    def _auto_detect_vibrant_rect(self, img):
        """内部方法：通过颜色特征自动检测小丑的鲜艳色区域"""
        # 转换为HSV空间，筛选高饱和度、高亮度的鲜艳色（小丑颜色共性）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation_thresh = 60  # 饱和度阈值（过滤灰色背景）
        value_thresh = 50       # 亮度阈值（过滤黑色背景）
        saturation_mask = hsv[:, :, 1] > saturation_thresh
        value_mask = hsv[:, :, 2] > value_thresh
        color_mask = np.bitwise_and(saturation_mask, value_mask).astype(np.uint8) * 255

        # 形态学优化：去除小噪点，填充区域小孔
        kernel = np.ones((7, 7), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找到最大连通块（小丑通常是图像中最大的鲜艳色区域）
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # 未检测到鲜艳色区域，返回None

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 适当扩大矩形，确保覆盖完整小丑区域
        expand = 10
        x = max(0, x - expand)
        y = max(0, y)
        w = min(img.shape[1] - x, w + expand)
        h = min(img.shape[0] - y, h + 2 * expand)

        return (x, y, w, h)

    def _grabcut_segment(self, img, rect):
        """内部方法：使用GrabCut算法分割指定矩形区域的前景（小丑）"""
        mask = np.zeros(img.shape[:2], np.uint8)  # 初始化掩码
        bgd_model = np.zeros((1, 65), np.float64)  # 背景模型
        fgd_model = np.zeros((1, 65), np.float64)  # 前景模型
        # 执行GrabCut（基于矩形初始化）
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
        # 生成最终掩码：确定前景(1)和可能前景(3)设为255，其他设为0
        final_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        return final_mask

    def _update_result_display(self):
        """内部方法：合并所有掩码并更新结果显示窗口"""
        if not self.masks:
            return  # 无掩码时跳过
        # 合并所有分割掩码（多次框选结果叠加）
        final_mask = np.zeros(self.current_img.shape[:2], np.uint8)
        for mask in self.masks:
            final_mask = cv2.bitwise_or(final_mask, mask)
        # 应用掩码显示当前结果（仅展示小丑区域，便于确认是否遗漏）
        result_img = cv2.bitwise_and(self.current_img, self.current_img, mask=final_mask)
        cv2.imshow(self.result_win_name, result_img)

    def _init_interactive_process(self, img):
        """内部方法：初始化交互式处理流程（自动检测+窗口创建）"""
        self.current_img = img.copy()
        # 1. 自动检测初始鲜艳色区域
        auto_rect = self._auto_detect_vibrant_rect(img)
        if auto_rect:
            print(f"Auto-detected area: {auto_rect}")
            initial_rect = auto_rect
        else:
            print("Auto-detect failed, use default area")
            # 自动检测失败时，使用图像中心区域作为默认初始框
            initial_rect = (
                int(img.shape[1] * 0.15), int(img.shape[0] * 0.15),
                int(img.shape[1] * 0.7), int(img.shape[0] * 0.7)
            )

        # 2. 创建窗口并设置鼠标回调
        cv2.namedWindow(self.select_win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.select_win_name, self._mouse_callback)
        # 显示初始框选窗口（绘制自动检测的矩形）
        init_display_img = img.copy()
        cv2.rectangle(
            init_display_img,
            (initial_rect[0], initial_rect[1]),
            (initial_rect[0] + initial_rect[2], initial_rect[1] + initial_rect[3]),
            (0, 255, 0), 2
        )
        cv2.imshow(self.select_win_name, init_display_img)

        # 3. 对初始区域执行GrabCut，初始化掩码列表
        initial_mask = self._grabcut_segment(img, initial_rect)
        self.masks.append(initial_mask)
        # 初始化结果显示窗口
        self._update_result_display()

    def rescue_clown(self, image):
        """
        对外接口：执行小丑剔除流程
        Args:
            image: 输入的原始彩色图像（OpenCV BGR格式，numpy.ndarray）
        Returns:
            numpy.ndarray | None: 剔除小丑后的图像（背景黑，保留气球+牵引线）；失败返回None
        """
        if image is None or len(image.shape) != 3:
            print("Error: Input must be a 3-channel BGR image")
            return None

        # 1. 初始化交互式流程
        self._init_interactive_process(image)

        # 2. 交互循环（等待用户确认或退出）
        while True:
            key = cv2.waitKey(0)
            if key == ord('f') or key == ord('F'):
                # 按F确认最终结果，执行小丑剔除
                print("Confirm final result, start rescuing clown...")
                break
            elif key == ord('q') or key == ord('Q'):
                # 按Q退出，释放资源
                print("Quit process")
                cv2.destroyAllWindows()
                return None

        # 3. 释放交互窗口
        cv2.destroyAllWindows()

        # 4. 合并所有掩码，生成最终小丑掩膜
        final_clown_mask = np.zeros(image.shape[:2], np.uint8)
        for mask in self.masks:
            final_clown_mask = cv2.bitwise_or(final_clown_mask, mask)

        # 5. 取反得到“去小丑掩膜”
        mask_remove_clown = cv2.bitwise_not(final_clown_mask)

        # === 新增：引入任务2的“前景掩膜”（黑底白前景）===
        from . import denoising
        foreground_mask = denoising.denoise(image)            # 二值：前景255 / 背景0
        foreground_mask = (foreground_mask > 0).astype('uint8') * 255  # 保底归一

        # === 关键：两掩膜相与，只留下“非小丑的前景”→ 仅气球+牵引线 ===
        final_mask = cv2.bitwise_and(foreground_mask, mask_remove_clown)

        # === 用最终掩膜抠原图，输出仍为“彩色” ===
        result = cv2.bitwise_and(image, image, mask=final_mask)
        return result

