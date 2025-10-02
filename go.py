import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGroupBox, QGridLayout, QSplitter
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
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGroupBox, QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class HSVImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.img = None
        self.processed_img = None
        self.setWindowTitle("枣红色字体提取工具（PyQt5版）")
        self.setGeometry(100, 100, 1200, 800)
        # 打开时自动最大化
        self.showMaximized()
        # 文件夹设置
        self.input_folder = "input_images"
        self.output_folder = "output_images"
        os.makedirs(self.output_folder, exist_ok=True)

        self.file_list = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ]
        if not self.file_list:
            raise FileNotFoundError("⚠️ input_images 文件夹中未找到图片！")

        self.img_index = 0
        self.output_mode = 0
        self.kernel = np.ones((3, 3), np.uint8)

        # HSV 默认参数
        self.hsv_params = {
            "H1_low": 0, "H1_high": 10,
            "S1_low": 80, "S1_high": 255,
            "V1_low": 80, "V1_high": 255,
            "H2_low": 170, "H2_high": 180,
            "S2_low": 80, "S2_high": 255,
            "V2_low": 80, "V2_high": 255,
        }

        self.img, self.hsv = self.load_image(self.img_index)
        self.update_processed_image()

        self.init_ui()

        # 定时刷新
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start()
    def update_preview(self):
        if self.img is None:
            return  # 没有图像时不更新
        orig_pix = self.cv2_to_qpixmap(self.img)
        processed_pix = self.cv2_to_qpixmap(self.processed_img)

        # ---- 根据窗口大小自适应缩放 ----
        available_width = self.orig_label.width()
        available_height = self.orig_label.height()

        orig_scaled = orig_pix.scaled(
            available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        processed_scaled = processed_pix.scaled(
            available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.orig_label.setPixmap(orig_scaled)
        self.processed_label.setPixmap(processed_scaled)
        self.img_name_label.setText(f"当前图片：{self.file_list[self.img_index]}")

    # 关键：当用户调整窗口大小时，强制刷新预览
    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

    def load_image(self, index):
        img_path = os.path.join(self.input_folder, self.file_list[index])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"❌ 无法读取图片：{img_path}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, hsv

    def process_image(self):
        lower1 = np.array([self.hsv_params["H1_low"], self.hsv_params["S1_low"], self.hsv_params["V1_low"]])
        upper1 = np.array([self.hsv_params["H1_high"], self.hsv_params["S1_high"], self.hsv_params["V1_high"]])
        lower2 = np.array([self.hsv_params["H2_low"], self.hsv_params["S2_low"], self.hsv_params["V2_low"]])
        upper2 = np.array([self.hsv_params["H2_high"], self.hsv_params["S2_high"], self.hsv_params["V2_high"]])

        mask1 = cv2.inRange(self.hsv, lower1, upper1)
        mask2 = cv2.inRange(self.hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        if self.output_mode == 0:
            cleaned = np.ones_like(self.img) * 255
            cleaned[mask > 0] = self.img[mask > 0]
        elif self.output_mode == 1:
            background = np.ones_like(self.img) * 255
            red_only = cv2.bitwise_and(self.img, self.img, mask=mask)
            cleaned = cv2.addWeighted(red_only, 1.0, background, 0.0, 0)
        else:
            cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return cleaned

    def update_processed_image(self):
        self.processed_img = self.process_image()

    def cv2_to_qpixmap(self, cv_img):
        if cv_img is None or cv_img.size == 0:
            return QPixmap()  # 返回一个空 pixmap，避免崩溃
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def update_preview(self):
        if self.img is None:
            return

        orig_pix = self.cv2_to_qpixmap(self.img)
        processed_pix = self.cv2_to_qpixmap(self.processed_img)

        # 获取两个 QLabel 的当前显示区域大小
        orig_size = self.orig_label.size()
        proc_size = self.processed_label.size()

        # 按 label 尺寸缩放，而不是按图片原始高度
        orig_scaled = orig_pix.scaled(
            orig_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        processed_scaled = processed_pix.scaled(
            proc_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.orig_label.setPixmap(orig_scaled)
        self.processed_label.setPixmap(processed_scaled)
        self.img_name_label.setText(f"当前图片：{self.file_list[self.img_index]}")


    def on_slider_change(self, slider_name, value):
        self.hsv_params[slider_name] = value
        self.param_labels[slider_name].setText(f"{value}")
        self.update_processed_image()

    def create_hsv_group(self, title, param_prefix):
        group = QGroupBox(title)
        layout = QGridLayout()

        if "H" in param_prefix:
            params = [(f"{param_prefix}low", (0, 180), f"{param_prefix.upper()}低："),
                      (f"{param_prefix}high", (0, 180), f"{param_prefix.upper()}高：")]
        else:
            params = [(f"{param_prefix}low", (0, 255), f"{param_prefix.upper()}低："),
                      (f"{param_prefix}high", (0, 255), f"{param_prefix.upper()}高：")]

        for row, (param_key, slider_range, label_text) in enumerate(params):
            label = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(*slider_range)
            slider.setValue(self.hsv_params[param_key])
            slider.valueChanged.connect(lambda v, k=param_key: self.on_slider_change(k, v))
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

        preview_layout = QHBoxLayout()
        self.orig_label = QLabel("原图")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.processed_label = QLabel("处理结果")
        self.processed_label.setAlignment(Qt.AlignCenter)

        splitter = QSplitter(Qt.Horizontal)
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

        params_layout.addLayout(group1)
        params_layout.addLayout(group2)
        main_layout.addLayout(params_layout)

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
        for idx, filename in enumerate(self.file_list):
            img_path = os.path.join(self.input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 跳过无法读取的图片：{filename}")
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower1 = np.array([self.hsv_params["H1_low"], self.hsv_params["S1_low"], self.hsv_params["V1_low"]])
            upper1 = np.array([self.hsv_params["H1_high"], self.hsv_params["S1_high"], self.hsv_params["V1_high"]])
            lower2 = np.array([self.hsv_params["H2_low"], self.hsv_params["S2_low"], self.hsv_params["V2_low"]])
            upper2 = np.array([self.hsv_params["H2_high"], self.hsv_params["S2_high"], self.hsv_params["V2_high"]])

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            mask = cv2.dilate(mask, self.kernel, iterations=1)

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
            save_path = os.path.join(self.output_folder, f"{name}_processed.png")
            cv2.imwrite(save_path, cleaned)
            print(f"✅ 已保存：{save_path}")
        print("🎉 批量保存完成！")


if __name__ == "__main__":
    app = QApplication([])
    editor = HSVImageEditor()
    editor.show()
    app.exec_()
