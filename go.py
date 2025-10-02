import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QSplitter,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


# ----------------- 参数说明 -----------------
# LowH1 / HighH1 : 第1段红色的色调范围（Hue）
#   - 建议范围：0–10
#   - 调节目标：捕捉靠近0°的红色
#
# LowS1 / HighS1 : 第1段红色的饱和度范围（Saturation）
#   - 建议范围：80–255
#   - 调节目标：去掉灰尘和浅色干扰，保留鲜艳红色
#
# LowV1 / HighV1 : 第1段红色的亮度范围（Value）
#   - 建议范围：80–255
#   - 调节目标：去掉背景上的暗点，保留较亮的红色文字
#
# LowH2 / HighH2 : 第2段红色的色调范围（Hue）
#   - 建议范围：170–180
#   - 调节目标：捕捉靠近180°的红色
#
# LowS2 / HighS2 : 第2段红色的饱和度范围（Saturation）
#   - 建议范围：80–255
#   - 调节目标：同上，控制是否保留灰红色
#
# LowV2 / HighV2 : 第2段红色的亮度范围（Value）
#   - 建议范围：80–255
#   - 调节目标：同上，控制明暗阈值
#
# -------------------------------------------
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QSplitter,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class HSVImageEditor(QMainWindow):
    def create_hue_preview_image(self, height=50, width=None):
        """生成HSV色调预览图：H从0到180渐变，S=255（最大饱和度），V=255（最大亮度）"""
        # 宽度不传则默认180
        width = width or 180

        # 1. 创建180像素宽HSV色条
        hsv_hue = np.zeros((height, 180, 3), dtype=np.uint8)
        for h in range(180):
            hsv_hue[:, h, 0] = h
            hsv_hue[:, h, 1] = 255
            hsv_hue[:, h, 2] = 255

        # 2. 转为BGR
        bgr_hue = cv2.cvtColor(hsv_hue, cv2.COLOR_HSV2BGR)

        # 3. 根据目标宽度缩放
        if width != 180:
            bgr_hue = cv2.resize(
                bgr_hue, (width, height), interpolation=cv2.INTER_LINEAR
            )

        # 4. 添加刻度
        step = 20
        for h in range(0, 181, step):
            x = int((h / 180) * bgr_hue.shape[1])
            cv2.line(bgr_hue, (x, height - 5), (x, height), (255, 255, 255), 1)
            cv2.putText(
                bgr_hue,
                str(h),
                (x - 5, height - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )

        return bgr_hue

    def __init__(self):
        super().__init__()

        # 先初始化属性，避免resizeEvent报错
        self.hue_preview_img = None
        self.img = None
        self.processed_img = None
        self.setWindowTitle("枣红色字体提取工具（PyQt5版）")
        # self.setGeometry(100, 100, 1200, 800)
        # 打开时自动最大化
        # self.showMaximized()
        # 文件夹设置
        self.input_folder = "input_images"
        self.output_folder = "output_images"
        os.makedirs(self.output_folder, exist_ok=True)

        self.file_list = [
            f
            for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ]
        if not self.file_list:
            raise FileNotFoundError("⚠️ input_images 文件夹中未找到图片！")

        self.img_index = 0
        self.output_mode = 0
        # self.kernel = np.ones((3, 3), np.uint8)
        # 形态学操作的核，用于去灰尘
        self.morph_kernel = np.ones((3, 3), np.uint8)
        # 最小连通区域面积，用于过滤小灰尘
        self.min_area = 50
        # HSV 默认参数
        self.hsv_params = {
            "H1_low": 0,
            "H1_high": 10,
            "S1_low": 80,
            "S1_high": 255,
            "V1_low": 80,
            "V1_high": 255,
            "H2_low": 170,
            "H2_high": 180,
            "S2_low": 80,
            "S2_high": 255,
            "V2_low": 80,
            "V2_high": 255,
        }

        self.img, self.hsv = self.load_image(self.img_index)
        self.update_processed_image()
        # -------------------------- 新增代码 --------------------------
        self.hue_preview_img = self.create_hue_preview_image().astype(
            np.uint8
        )  # 生成H色调光谱图
        assert self.hue_preview_img.dtype == np.uint8, "图像数据类型错误！应为np.uint8"
        # --------------------------------------------------------------
        self.init_ui()

        # 定时刷新
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start()
        self.update_preview()  # 添加此行确保初始加载时显示预览条
        self.update_hue_preview()

    def update_hue_preview(self):
        if self.hue_preview_img is None:
            return
        rgb_img = cv2.cvtColor(self.hue_preview_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 缩放到Label宽度自适应
        pixmap = pixmap.scaled(
            self.hue_preview_label.width(),
            self.hue_preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.hue_preview_label.setPixmap(pixmap)

    def update_preview(self):
        if self.img is None:
            return  # 没有图像时不更新
        orig_pix = self.cv2_to_qpixmap(self.img)
        processed_pix = self.cv2_to_qpixmap(self.processed_img)

        # ---- 根据窗口大小自适应缩放 ----
        available_width = self.orig_label.width()
        available_height = self.orig_label.height()

        orig_scaled = orig_pix.scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        processed_scaled = processed_pix.scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.orig_label.setPixmap(orig_scaled)
        self.processed_label.setPixmap(processed_scaled)
        self.img_name_label.setText(f"当前图片：{self.file_list[self.img_index]}")

    # 关键：当用户调整窗口大小时，强制刷新预览
    def resizeEvent(self, event):
        # 根据 Label 宽度重新生成色条
        if hasattr(self, "hue_preview_label") and self.hue_preview_label is not None:
            label_width = max(self.hue_preview_label.width(), 100)  # 避免太小
            self.hue_preview_img = self.create_hue_preview_image(width=label_width)
        self.update_preview()
        self.update_hue_preview()
        super().resizeEvent(event)

    def load_image(self, index):
        img_path = os.path.join(self.input_folder, self.file_list[index])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"❌ 无法读取图片：{img_path}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, hsv

    def process_image(self):
        # 显示繁忙鼠标图标
        QApplication.setOverrideCursor(Qt.WaitCursor)
        lower1 = np.array(
            [
                self.hsv_params["H1_low"],
                self.hsv_params["S1_low"],
                self.hsv_params["V1_low"],
            ]
        )
        upper1 = np.array(
            [
                self.hsv_params["H1_high"],
                self.hsv_params["S1_high"],
                self.hsv_params["V1_high"],
            ]
        )
        lower2 = np.array(
            [
                self.hsv_params["H2_low"],
                self.hsv_params["S2_low"],
                self.hsv_params["V2_low"],
            ]
        )
        upper2 = np.array(
            [
                self.hsv_params["H2_high"],
                self.hsv_params["S2_high"],
                self.hsv_params["V2_high"],
            ]
        )

        mask1 = cv2.inRange(self.hsv, lower1, upper1)
        mask2 = cv2.inRange(self.hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学开运算，先腐蚀后膨胀，去除小灰尘
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        # 闭运算，填充文字内部的小缺口
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)
        # 连通区域过滤，去除小面积区域（灰尘）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > self.min_area:
                filtered_mask[labels == i] = 255
        mask = filtered_mask
        if self.output_mode == 0:
            cleaned = np.ones_like(self.img) * 255
            cleaned[mask > 0] = self.img[mask > 0]
        elif self.output_mode == 1:
            background = np.ones_like(self.img) * 255
            red_only = cv2.bitwise_and(self.img, self.img, mask=mask)
            cleaned = cv2.addWeighted(red_only, 1.0, background, 0.0, 0)
        else:
            cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # 恢复默认鼠标图标
        QApplication.restoreOverrideCursor()
        return cleaned

    def update_processed_image(self):
        self.processed_img = self.process_image()

    def cv2_to_qpixmap(self, cv_img):
        if cv_img is None or cv_img.size == 0 or cv_img.dtype != np.uint8:
            return QPixmap()
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 关键：BGR转RGB
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def on_slider_value_update(self, slider_name, value):
        self.hsv_params[slider_name] = value  # 同步参数值
        self.param_labels[slider_name].setText(f"{value}")  # 实时更新数值显示

    # 2. 滑块释放时才处理图像（耗时操作，仅触发1次）
    def on_slider_release(self, slider_name):
        self.update_processed_image()  # 执行图像处理+预览更新

    # def on_slider_change(self, slider_name, value):
    #    self.hsv_params[slider_name] = value
    #    self.param_labels[slider_name].setText(f"{value}")
    #    self.update_processed_image()

    def create_hsv_group(self, title, param_prefix):
        group = QGroupBox(title)
        layout = QGridLayout()

        if "H" in param_prefix:
            params = [
                (f"{param_prefix}low", (0, 180), f"{param_prefix.upper()}低："),
                (f"{param_prefix}high", (0, 180), f"{param_prefix.upper()}高："),
            ]
        else:
            params = [
                (f"{param_prefix}low", (0, 255), f"{param_prefix.upper()}低："),
                (f"{param_prefix}high", (0, 255), f"{param_prefix.upper()}高："),
            ]

        for row, (param_key, slider_range, label_text) in enumerate(params):
            label = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*slider_range)
            slider.setValue(self.hsv_params[param_key])
            # 新逻辑（新增）：
            # - 拖动时触发：仅更新数值
            slider.valueChanged.connect(
                lambda v, k=param_key: self.on_slider_value_update(k, v)
            )
            # - 释放时触发：处理图像
            slider.sliderReleased.connect(lambda k=param_key: self.on_slider_release(k))
            value_label = QLabel(str(self.hsv_params[param_key]))
            value_label.setAlignment(Qt.AlignRight)
            value_label.setFixedWidth(50)
            self.param_labels[param_key] = value_label

            layout.addWidget(label, row, 0)
            layout.addWidget(slider, row, 1)
            layout.addWidget(value_label, row, 2)

        group.setLayout(layout)
        return group

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.img_name_label = QLabel(f"当前图片：{self.file_list[self.img_index]}")
        self.img_name_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.img_name_label)
        # -------------------------- 新增代码：H色调预览条布局 --------------------------
        hue_layout = QHBoxLayout()
        hue_title_label = QLabel("HSV色调（H）预览（0-180）：")  # 预览条标题
        self.hue_preview_label = QLabel()  # 用于显示H色调光谱图的Label
        self.hue_preview_label.setMinimumHeight(50)  # 高度固定
        # 不设置固定宽度，让它自适应父布局
        self.hue_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hue_layout.addWidget(hue_title_label)
        hue_layout.addWidget(
            self.hue_preview_label, stretch=1
        )  # 预览条占满剩余宽度（自适应窗口）
        self.hue_preview_label.setFixedHeight(50)
        main_layout.addLayout(hue_layout)

        # ------------------------------------------------------------------------------------------
        preview_layout = QVBoxLayout()
        self.orig_label = QLabel("原图")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.processed_label = QLabel("处理结果")
        self.processed_label.setAlignment(Qt.AlignCenter)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.orig_label)
        splitter.addWidget(self.processed_label)
        splitter.setSizes([600, 600])
        preview_layout.addWidget(splitter)
        main_layout.addLayout(preview_layout, stretch=1)

        self.param_labels = {}
        params_layout = QHBoxLayout()

        group1 = QVBoxLayout()
        group1.addWidget(self.create_hsv_group("红色区间1 - H", "H1_"))
        group1.addWidget(self.create_hsv_group("红色区间1 - S", "S1_"))
        group1.addWidget(self.create_hsv_group("红色区间1 - V", "V1_"))

        group2 = QVBoxLayout()
        group2.addWidget(self.create_hsv_group("红色区间2 - H", "H2_"))
        group2.addWidget(self.create_hsv_group("红色区间2 - S", "S2_"))
        group2.addWidget(self.create_hsv_group("红色区间2 - V", "V2_"))

        # 添加去灰尘相关的控制
        dust_layout = QVBoxLayout()
        dust_group = QGroupBox("去灰尘设置")
        dust_inner_layout = QGridLayout()

        # 形态学核大小滑块
        kernel_size_label = QLabel("形态学核大小：")
        self.kernel_size_slider = QSlider(Qt.Horizontal)
        self.kernel_size_slider.setRange(1, 10)
        self.kernel_size_slider.setValue(3)
        self.kernel_size_slider.sliderReleased.connect(self.on_kernel_size_release)
        kernel_size_value_label = QLabel("3")
        kernel_size_value_label.setAlignment(Qt.AlignRight)
        kernel_size_value_label.setFixedWidth(50)
        self.kernel_size_value_label = kernel_size_value_label

        # 最小连通面积滑块
        min_area_label = QLabel("最小连通面积：")
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(10, 200)
        self.min_area_slider.setValue(self.min_area)
        self.min_area_slider.sliderReleased.connect(self.on_min_area_release)
        min_area_value_label = QLabel(str(self.min_area))
        min_area_value_label.setAlignment(Qt.AlignRight)
        min_area_value_label.setFixedWidth(50)
        self.min_area_value_label = min_area_value_label

        dust_inner_layout.addWidget(kernel_size_label, 0, 0)
        dust_inner_layout.addWidget(self.kernel_size_slider, 0, 1)
        dust_inner_layout.addWidget(kernel_size_value_label, 0, 2)
        dust_inner_layout.addWidget(min_area_label, 1, 0)
        dust_inner_layout.addWidget(self.min_area_slider, 1, 1)
        dust_inner_layout.addWidget(min_area_value_label, 1, 2)

        dust_group.setLayout(dust_inner_layout)
        dust_layout.addWidget(dust_group)

        params_layout.addLayout(group1)
        params_layout.addLayout(group2)
        params_layout.addLayout(dust_layout)  # 将dust_layout添加到params_layout
        main_layout.addLayout(params_layout)  # 将params_layout添加到main_layout

        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张（←）")
        self.prev_btn.clicked.connect(lambda: self.switch_image(-1))
        self.next_btn = QPushButton("下一张（→）")
        self.next_btn.clicked.connect(lambda: self.switch_image(1))
        self.mode_btn = QPushButton("切换模式（当前：白底红字）")
        self.mode_btn.clicked.connect(self.switch_mode)
        self.save_btn = QPushButton("批量保存所有图片")
        self.save_btn.clicked.connect(self.batch_save)
        self.quit_btn = QPushButton("退出")
        self.quit_btn.clicked.connect(QApplication.quit)

        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addWidget(self.mode_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.quit_btn)
        main_layout.addLayout(btn_layout)

    # 1. 形态学核大小滑块：释放时处理
    def on_kernel_size_release(self):
        value = self.kernel_size_slider.value()  # 释放后获取最终值
        self.morph_kernel = np.ones((value, value), np.uint8)  # 更新核大小
        self.kernel_size_value_label.setText(str(value))  # 更新数值显示
        self.update_processed_image()  # 处理图像

    # 2. 最小连通面积滑块：释放时处理
    def on_min_area_release(self):
        value = self.min_area_slider.value()  # 释放后获取最终值
        self.min_area = value  # 更新面积阈值
        self.min_area_value_label.setText(str(value))  # 更新数值显示
        self.update_processed_image()  # 处理图像

    def switch_image(self, step):
        self.img_index = (self.img_index + step) % len(self.file_list)
        self.img, self.hsv = self.load_image(self.img_index)
        self.update_processed_image()
        print(f"🔄 切换到：{self.file_list[self.img_index]}")

    def switch_mode(self):
        self.output_mode = (self.output_mode + 1) % 3
        mode_names = ["白底红字", "叠加模式", "掩码模式"]
        self.mode_btn.setText(f"切换模式（当前：{mode_names[self.output_mode]}）")
        self.processed_label.setText(f"处理结果（{mode_names[self.output_mode]}）")
        self.update_processed_image()

    def batch_save(self):
        print("📤 开始批量保存...")
        # 显示繁忙鼠标图标
        QApplication.setOverrideCursor(Qt.WaitCursor)
        for idx, filename in enumerate(self.file_list):
            img_path = os.path.join(self.input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 跳过无法读取的图片：{filename}")
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower1 = np.array(
                [
                    self.hsv_params["H1_low"],
                    self.hsv_params["S1_low"],
                    self.hsv_params["V1_low"],
                ]
            )
            upper1 = np.array(
                [
                    self.hsv_params["H1_high"],
                    self.hsv_params["S1_high"],
                    self.hsv_params["V1_high"],
                ]
            )
            lower2 = np.array(
                [
                    self.hsv_params["H2_low"],
                    self.hsv_params["S2_low"],
                    self.hsv_params["V2_low"],
                ]
            )
            upper2 = np.array(
                [
                    self.hsv_params["H2_high"],
                    self.hsv_params["S2_high"],
                    self.hsv_params["V2_high"],
                ]
            )

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

            # 应用形态学操作和连通区域过滤（与实时处理一致）
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1
            )
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            filtered_mask = np.zeros_like(mask)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > self.min_area:
                    filtered_mask[labels == i] = 255
            mask = filtered_mask

            if self.output_mode == 0:
                cleaned = np.ones_like(img) * 255
                cleaned[mask > 0] = img[mask > 0]
            elif self.output_mode == 1:
                background = np.ones_like(img) * 255
                red_only = cv2.bitwise_and(img, img, mask=mask)
                cleaned = cv2.addWeighted(red_only, 1.0, background, 0.0, 0)
            else:
                cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            name, _ = os.path.splitext(filename)
            save_path = os.path.join(self.output_folder, f"{name}.png")
            cv2.imwrite(save_path, cleaned)
            print(f"✅ 已保存：{save_path}")
        print("🎉 批量保存完成！")
        # 恢复默认鼠标图标
        QApplication.restoreOverrideCursor()


if __name__ == "__main__":
    app = QApplication([])
    editor = HSVImageEditor()
    editor.showMaximized()
    app.exec_()
