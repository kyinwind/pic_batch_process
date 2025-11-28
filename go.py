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
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent, QResizeEvent
from typing import Optional

try:
    import torch
    import kornia as K

    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_GPU_AVAILABLE = False

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
    QComboBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent
from typing import Optional


class HSVImageEditor(QMainWindow):
    # 重写键盘按下事件
    def keyPressEvent(self, a0: Optional[QKeyEvent]):
        if a0 is None:
            return
        key = a0.key()
        mod = a0.modifiers()
        # print(f"按下键：{key}, 修饰符：{mod}")
        # 处理 + 号（包括小键盘 +）
        if key == Qt.Key.Key_Plus or (
            key == Qt.Key.Key_Equal and mod == Qt.KeyboardModifier.ShiftModifier
        ):
            print(f"按下+键：{key}, 修饰符：{mod}")
            if self.h1_high_slider:
                current = self.h1_high_slider.value()
                if current < 180:
                    self.h1_high_slider.setValue(current + 1)
                    self.on_slider_value_update("H1_high", current + 1)
                    self.on_slider_release("H1_high")
            return  # 阻止继续传递

        # 处理 - 号（包括小键盘 -）
        elif key == Qt.Key.Key_Minus:
            print(f"按下-键：{key}, 修饰符：{mod}")
            if self.h1_high_slider:
                current = self.h1_high_slider.value()
                if current > 0:
                    self.h1_high_slider.setValue(current - 1)
                    self.on_slider_value_update("H1_high", current - 1)
                    self.on_slider_release("H1_high")
            return

        # 其他按键交给父类处理
        super().keyPressEvent(a0)

    def on_image_click(self, event, label):
        if self.img is None:
            return

        # --- 左键：设置放大中心 ---
        if event.button() == Qt.MouseButton.LeftButton:
            pixmap = label.pixmap()
            if pixmap is None:
                return

            x, y = event.pos().x(), event.pos().y()
            pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
            label_w, label_h = label.width(), label.height()

            # 计算偏移量（居中时的空白边）
            offset_x = (label_w - pixmap_w) // 2
            offset_y = (label_h - pixmap_h) // 2

            # 如果点击在空白处，忽略
            if not (
                offset_x <= x < offset_x + pixmap_w
                and offset_y <= y < offset_y + pixmap_h
            ):
                return

            # 转换为原图坐标
            rel_x = (x - offset_x) / pixmap_w
            rel_y = (y - offset_y) / pixmap_h

            img_h, img_w = self.img.shape[:2]
            self.zoom_center = (int(rel_x * img_w), int(rel_y * img_h))

            print(f"🔍 放大中心更新为: {self.zoom_center}")
            self.update_preview()

        # --- 右键：用系统默认程序打开图片 ---
        elif event.button() == Qt.MouseButton.RightButton:

            if label == self.orig_label:

                """双击原图时，用系统默认程序打开原图文件"""
                if not self.file_list:
                    self.show_toast("没有图片可打开")
                    return

                # 获取原图路径
                orig_path = os.path.join(
                    self.input_folder, self.file_list[self.img_index]
                )
                if os.path.exists(orig_path):
                    try:
                        # 用系统默认程序打开文件
                        print(f"orig_path: {orig_path}")
                        os.startfile(orig_path)
                    except Exception as e:
                        self.show_toast(f"打开失败：{str(e)}")
                else:
                    self.show_toast(f"原图不存在：{orig_path}")
            elif label == self.processed_label:
                """双击处理结果时，用系统默认程序打开处理后的图片"""
                if not self.file_list or self.processed_img is None:
                    self.show_toast("没有处理结果可打开")
                    return
                # 生成处理后图片的保存路径（与保存当前图片的路径一致）
                filename = self.file_list[self.img_index]
                name, _ = os.path.splitext(filename)
                processed_path = os.path.join(self.output_folder, f"{name}.png")
                print(f"processed_path: {processed_path}")
                # 检查文件是否存在
                if os.path.exists(processed_path):
                    try:
                        os.startfile(processed_path)
                    except Exception as e:
                        self.show_toast(f"打开失败：{str(e)}")
                else:
                    self.show_toast("处理结果未保存，请先保存图片")
            else:
                return

    def show_toast(self, message):
        """显示一个短暂的提示窗口"""
        # 创建提示标签
        toast = QLabel(message, self)
        # 设置样式：黑色半透明背景、白色文字、居中
        toast.setStyleSheet(
            """
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
        """
        )
        # 设置字体
        font = QFont()
        font.setPointSize(10)
        toast.setFont(font)
        # 设置对齐方式
        toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 调整大小
        toast.adjustSize()
        # 放置在窗口底部中间
        toast.move(
            (self.width() - toast.width()) // 2,
            30,  # 底部留出30px间距
        )
        # 显示提示
        toast.show()
        # 3秒后自动关闭
        QTimer.singleShot(3000, toast.deleteLater)

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

    def save_current(self):
        print("💾 保存当前图片...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # 取当前处理后的图像
        cleaned = self.processed_img.copy()  # type: ignore

        # 生成保存路径
        filename = self.file_list[self.img_index]
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(self.output_folder, f"{name}.png")

        # 保存文件
        success = cv2.imwrite(save_path, cleaned)
        if success:
            print(f"✅ 已保存当前图片：{save_path}")
            # 显示保存成功的toast提示
            self.show_toast(f"保存成功：{os.path.basename(save_path)}")
        else:
            print(f"❌ 保存失败：{save_path}")
            self.show_toast(f"保存失败：{os.path.basename(save_path)}")

        QApplication.restoreOverrideCursor()

    def __init__(self):
        super().__init__()

        # 先初始化属性，避免resizeEvent报错
        self.hue_preview_img = None
        self.img = None
        self.processed_img = None
        self.setWindowTitle("枣红色字体提取工具（PyQt5版）")
        # 初始化时获取H1_high滑块（需要先在init_ui中标记滑块）
        self.h1_high_slider = None  # 用于存储H1_high滑块的引用
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

        # ✅ 新增：高斯模糊核大小
        self.gaussian_kernel_size = 3
        # ✅ 新增：局部放大图参数
        self.zoom_size = 600  # 放大图尺寸（像素）
        self.zoom_factor = 3  # 放大倍数
        self.zoom_center = None  # 默认没有，表示用中心点

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
        # 1. 先加载图像
        self.img, self.hsv = self.load_image(self.img_index)

        # 2. 先初始化UI
        self.init_ui()
        # 3. 生成色调预览图
        self.hue_preview_img = self.create_hue_preview_image().astype(
            np.uint8
        )  # 生成H色调光谱图
        assert self.hue_preview_img.dtype == np.uint8, "图像数据类型错误！应为np.uint8"
        # 4. 最后处理图像
        self.update_processed_image()
        self.update_preview()
        self.update_hue_preview()
        # 定时刷新
        self.timer = QTimer()
        self.timer.setInterval(500)
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
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.hue_preview_label.setPixmap(pixmap)

    def update_preview(self):
        if self.img is None:
            return  # 没有图像时不更新

        # --- 左边原图和处理图 ---
        orig_pix = self.cv2_to_qpixmap(self.img)
        processed_pix = self.cv2_to_qpixmap(self.processed_img)

        available_width = self.orig_label.width()
        available_height = self.orig_label.height()

        orig_scaled = orig_pix.scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        processed_scaled = processed_pix.scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.orig_label.setPixmap(orig_scaled)
        self.processed_label.setPixmap(processed_scaled)
        self.img_name_label.setText(f"当前图片：{self.file_list[self.img_index]}")

        # --- 动态调整右侧放大图大小 ---
        # total_left_height = self.orig_label.height() + self.processed_label.height()
        # self.zoom_size = total_left_height // 2 - 20  # 每张放大图大约占一半高度
        right_height = self.zoom_orig_label.height()
        self.zoom_size = right_height - 20
        # screen = QApplication.primaryScreen()
        # screen_height = screen.size().height() if screen else 1080
        # if screen_height > 3000:
        #     max_zoom_size = 600
        # else:
        #     max_zoom_size = 450
        # self.zoom_size = min(right_height, max_zoom_size)  # 确保不超过可用高度
        # --- 计算放大中心 ---
        h, w = self.img.shape[:2]
        if self.zoom_center is None:
            center_x, center_y = w // 2, h // 2  # 默认取中心点
        else:
            center_x, center_y = self.zoom_center

        half = self.zoom_size // (2 * self.zoom_factor)

        # 确保不越界
        x1, y1 = max(center_x - half, 0), max(center_y - half, 0)
        x2, y2 = min(center_x + half, w), min(center_y + half, h)

        roi_orig = self.img[y1:y2, x1:x2]
        if self.processed_img is None:
            return  # 或 raise Exception("处理图像未生成")
        roi_proc = self.processed_img[y1:y2, x1:x2]

        if roi_orig.size == 0 or roi_proc.size == 0:
            return  # 无效区域

        # --- 放大并显示 ---
        zoomed_orig = cv2.resize(
            roi_orig, (self.zoom_size, self.zoom_size), interpolation=cv2.INTER_CUBIC
        )
        zoomed_proc = cv2.resize(
            roi_proc, (self.zoom_size, self.zoom_size), interpolation=cv2.INTER_CUBIC
        )

        self.zoom_orig_label.setPixmap(self.cv2_to_qpixmap(zoomed_orig))
        self.zoom_processed_label.setPixmap(self.cv2_to_qpixmap(zoomed_proc))

    # 关键：当用户调整窗口大小时，强制刷新预览
    def resizeEvent(self, a0: Optional[QResizeEvent]):
        if a0 is None:
            return
        # 根据 Label 宽度重新生成色条
        if hasattr(self, "hue_preview_label") and self.hue_preview_label is not None:
            label_width = max(self.hue_preview_label.width(), 100)  # 避免太小
            self.hue_preview_img = self.create_hue_preview_image(width=label_width)
        self.update_preview()
        self.update_hue_preview()
        super().resizeEvent(a0)

    def load_image(self, index):
        img_path = os.path.join(self.input_folder, self.file_list[index])
        img: np.ndarray = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"❌ 无法读取图片：{img_path}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, hsv

    def process_image(self, img: Optional[np.ndarray] = None):
        """
        Dispatch: 如果检测到 PyTorch+CUDA 则使用 GPU 实现（via Kornia），否则使用原有 CPU 实现（保留原逻辑）。
        这样最小改动即可在有合适环境时自动使用 NVIDIA GPU。
        """
        # 如果传入 img，直接使用传入的，否则用 self.img（与原逻辑一致）
        target_img = img if img is not None else self.img

        if target_img is None:
            return target_img

        # 优先使用 PyTorch+Kornia 在 GPU 上运行（若可用）
        if TORCH_GPU_AVAILABLE:
            try:
                # 将 BGR numpy 转为 RGB torch.Tensor, float32, 0..1，然后上 GPU
                rgb = target_img[..., ::-1].copy()  # BGR->RGB
                t = (
                    torch.from_numpy(rgb)
                    .to(dtype=torch.float32)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                t = t.cuda()

                # 1) RGB -> HSV (Kornia 返回 H in [0,1] representing 0..360°, S,V in [0,1])
                hsv = K.color.rgb_to_hsv(t)

                # 2) 构造掩码：将你的参数从原范围映射到 kornia 的范围
                # H：原来 0..180 -> 0..1  映射: h/180
                # S,V：原来 0..255 -> 0..1  映射: s/255, v/255
                H = hsv[:, 0:1, :, :]
                S = hsv[:, 1:2, :, :]
                V = hsv[:, 2:3, :, :]

                # 构建两段阈值掩码（注意 H 的环绕 0/180 情况）
                def hsv_mask_range(h_low, h_high, s_low, s_high, v_low, v_high):
                    h_low_f = h_low / 180.0
                    h_high_f = h_high / 180.0
                    s_low_f = s_low / 255.0
                    s_high_f = s_high / 255.0
                    v_low_f = v_low / 255.0
                    v_high_f = v_high / 255.0

                    if h_low_f <= h_high_f:
                        mh = (H >= h_low_f) & (H <= h_high_f)
                    else:
                        # 环绕情况，例如 170..180 and 0..10 会被分成两段处理（这里直接用或）
                        mh = (H >= h_low_f) | (H <= h_high_f)

                    ms = (S >= s_low_f) & (S <= s_high_f)
                    mv = (V >= v_low_f) & (V <= v_high_f)
                    return (mh & ms & mv).to(dtype=torch.float32)

                m1 = hsv_mask_range(
                    self.hsv_params["H1_low"],
                    self.hsv_params["H1_high"],
                    self.hsv_params["S1_low"],
                    self.hsv_params["S1_high"],
                    self.hsv_params["V1_low"],
                    self.hsv_params["V1_high"],
                )
                m2 = hsv_mask_range(
                    self.hsv_params["H2_low"],
                    self.hsv_params["H2_high"],
                    self.hsv_params["S2_low"],
                    self.hsv_params["S2_high"],
                    self.hsv_params["V2_low"],
                    self.hsv_params["V2_high"],
                )
                mask = ((m1 + m2) > 0).to(dtype=torch.float32)  # 0/1 mask float

                # 可选：高斯模糊（Kornia 提供 gaussian blur）
                if self.gaussian_checkbox.isChecked():
                    # ksize 必须为奇数，这里根据 self.gaussian_kernel_size
                    gks = self.gaussian_kernel_size
                    mask = K.filters.gaussian_blur2d(mask, (gks, gks), (0.0, 0.0))

                # 可选：形态学去噪（Kornia 有形态学函数）
                if self.dust_checkbox.isChecked():
                    # 先二值化并做开/闭运算（迭代次数可调整）
                    bin_mask = (mask > 0.5).to(dtype=torch.float32)
                    # 这里用腐蚀/膨胀近似开/闭
                    kern = torch.ones(
                        (1, 1, self.morph_kernel.shape[0], self.morph_kernel.shape[1]),
                        device=bin_mask.device,
                    )
                    # 你可以用 kornia.morphology.erode/dilate if available; 下为示意：
                    bin_mask = K.morphology.dilation(bin_mask, kern)
                    bin_mask = K.morphology.erosion(bin_mask, kern)
                    mask = bin_mask

                # 根据 output_mode 生成 cleaned（在 GPU 上）
                if self.output_mode == 0:
                    # 白底红字：把 rgb * mask + white*(1-mask)
                    colored = t * mask
                    white = torch.ones_like(colored)
                    cleaned_t = colored + white * (1.0 - mask)
                elif self.output_mode == 1:
                    # 叠加模式：直接把红色通道保留到背景
                    red_only = t * mask
                    cleaned_t = red_only + torch.ones_like(t) * 0.0  # adjust as needed
                else:
                    # 掩码模式：把 mask 转成三通道灰度
                    cleaned_t = torch.cat([mask, mask, mask], dim=1)

                # 转回 numpy BGR uint8
                cleaned_t = (
                    (cleaned_t.clamp(0.0, 1.0) * 255.0)
                    .to(dtype=torch.uint8)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                cleaned_bgr = cleaned_t[..., ::-1]  # RGB->BGR
                return cleaned_bgr

            except Exception as e:
                # 若 GPU 路径失败则回退到 CPU 路径（并打印原因）
                print("GPU processing failed, fallback to CPU. Reason:", e)

        # 回退：调用原有 CPU 实现（保留你原先的代码逻辑）
        # 注意：这里假定你的原 process_image CPU 逻辑已移动到一个方法 process_image_cpu
        try:
            return self.process_image_cpu(img)
        except Exception:
            # 如果没有拆分函数（原实现在本函数），直接保持原实现（或者复制原实现代码到 process_image_cpu）
            return None

    def process_image_cpu(self, img: Optional[np.ndarray] = None):
        # 如果传入了 img，则用传入的；否则用 self.img（兼容实时处理）
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            img = self.img
            hsv = self.hsv

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

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

        # 颜色区间掩码
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # ✅ 判断是否启用去灰尘
        if self.dust_checkbox.isChecked():
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1
            )

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            filtered_mask = np.zeros_like(mask)
            for i in range(1, num_labels):  # 跳过背景
                if stats[i, cv2.CC_STAT_AREA] > self.min_area:
                    filtered_mask[labels == i] = 255
            mask = filtered_mask
        # ✅ 判断是否启用高斯模糊
        if self.gaussian_checkbox.isChecked():
            # --- 高斯模糊平滑边缘 ---
            print(self.gaussian_kernel_size)
            mask = cv2.GaussianBlur(
                mask, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0
            )

        # --- 锐化处理 ---

        # if self.sharpen_checkbox.isChecked():
        #     # 方法1：卷积核锐化
        #     # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #     # mask = cv2.filter2D(mask, -1, kernel)
        #     # 方法2锐化处理：Unsharp Mask
        #     blurred = cv2.GaussianBlur(mask, (3, 3), 0)
        #     mask = cv2.addWeighted(mask, 2.0, blurred, -1.0, 0)

        # 根据输出模式生成结果
        if self.output_mode == 0:
            cleaned = np.ones_like(img) * 255
            cleaned[mask > 0] = img[mask > 0]
        elif self.output_mode == 1:
            background = np.ones_like(img) * 255
            red_only = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            cleaned = cv2.addWeighted(red_only, 1.0, background, 0.0, 0)
        else:
            cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

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
    # 3. 新增：高斯模糊核大小滑块释放时处理
    def on_gaussian_kernel_size_value_change(self):
        self.gaussian_kernel_size = self.gaussian_kernel_size_slider.value()
        self.gaussian_kernel_size_value_label.setText(
            str(self.gaussian_kernel_size_slider.value())
        )

    def on_gaussian_kernel_size_release(self):
        self.gaussian_kernel_size_value_label.setText(
            str(self.gaussian_kernel_size_slider.value())
        )
        if self.gaussian_kernel_size_slider.value() % 2 == 1:
            if self.gaussian_checkbox.isChecked():
                self.gaussian_kernel_size = self.gaussian_kernel_size_slider.value()
                self.update_processed_image()
        else:
            self.show_toast("高斯模糊核大小必须为奇数")

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
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(*slider_range)
            slider.setValue(self.hsv_params[param_key])

            # 关键：标记H1_high滑块，方便后续获取
            if param_key == "H1_high":
                self.h1_high_slider = slider  # 保存滑块引用
            # 新逻辑（新增）：
            # - 拖动时触发：仅更新数值
            slider.valueChanged.connect(
                lambda v, k=param_key: self.on_slider_value_update(k, v)
            )
            # - 释放时触发：处理图像
            slider.sliderReleased.connect(lambda k=param_key: self.on_slider_release(k))
            value_label = QLabel(str(self.hsv_params[param_key]))
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_label.setFixedWidth(50)
            self.param_labels[param_key] = value_label

            layout.addWidget(label, row, 0)
            layout.addWidget(slider, row, 1)
            layout.addWidget(value_label, row, 2)

        group.setLayout(layout)
        return group

    # ----------------- 新增方法 -----------------
    def on_image_selected(self, index):
        """当下拉框选择图片时切换"""
        self.img_index = index
        self.img, self.hsv = self.load_image(self.img_index)
        self.update_processed_image()
        self.update_preview()
        print(f"📂 已选择：{self.file_list[self.img_index]}")

    # --------------------------------------------
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        # ---------------- 图片选择下拉框 ----------------
        select_layout = QHBoxLayout()
        self.img_combo = QComboBox()
        self.img_combo.addItems(self.file_list)  # 加载所有图片名称
        self.img_combo.setCurrentIndex(self.img_index)
        self.img_combo.currentIndexChanged.connect(self.on_image_selected)
        select_layout.addWidget(QLabel("选择图片："))
        select_layout.addWidget(self.img_combo, stretch=1)

        main_layout.addLayout(select_layout)
        # -------------------------------------------------
        self.img_name_label = QLabel(f"当前图片：{self.file_list[self.img_index]}")
        self.img_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        # preview_layout = QVBoxLayout()
        # self.orig_label = QLabel("原图")
        # self.orig_label.setAlignment(Qt.AlignCenter)
        # self.orig_label.mouseDoubleClickEvent = (
        #     self.on_orig_image_dblclick
        # )  # 绑定双击事件
        # self.processed_label = QLabel("处理结果")
        # self.processed_label.setAlignment(Qt.AlignCenter)
        # self.processed_label.mouseDoubleClickEvent = (
        #     self.on_processed_image_dblclick
        # )  # 绑定双击事件

        # splitter = QSplitter(Qt.Vertical)
        # splitter.addWidget(self.orig_label)
        # splitter.addWidget(self.processed_label)
        # splitter.setSizes([600, 600])
        # preview_layout.addWidget(splitter)
        # main_layout.addLayout(preview_layout, stretch=1)

        # -------------------------- 新增代码：局部放大图布局 --------------------------
        # ---------------- 主图与局部放大图布局 ----------------
        preview_container = QHBoxLayout()
        preview_container.setSpacing(5)
        # 左侧：主图（原图 + 处理图）
        main_view_layout = QVBoxLayout()
        self.orig_label = QLabel("原图")
        self.orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.processed_label = QLabel("处理结果")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.orig_label.mousePressEvent = lambda e: self.on_image_click(
            e, self.orig_label
        )
        self.processed_label.mousePressEvent = lambda e: self.on_image_click(
            e, self.processed_label
        )

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.orig_label)
        splitter.addWidget(self.processed_label)
        splitter.setSizes([600, 600])
        main_view_layout.addWidget(splitter)

        # 右侧：局部放大图
        zoom_layout = QVBoxLayout()
        self.zoom_orig_label = QLabel("原图\n(中心放大×5)")
        self.zoom_orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_orig_label.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        self.zoom_orig_label.setFixedWidth(self.zoom_size + 20)
        # self.zoom_orig_label.setFixedHeight(self.zoom_size + 40)
        self.zoom_orig_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.zoom_processed_label = QLabel("处理图\n(中心放大×5)")
        self.zoom_processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_processed_label.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        self.zoom_processed_label.setFixedWidth(self.zoom_size + 20)
        # self.zoom_processed_label.setFixedHeight(self.zoom_size + 40)
        self.zoom_processed_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        zoom_layout.addWidget(self.zoom_orig_label)
        zoom_layout.addWidget(self.zoom_processed_label)

        # 组合左右
        preview_container.addLayout(main_view_layout, stretch=3)
        preview_container.addLayout(zoom_layout, stretch=1)  # 左右比例 3:1

        main_layout.addLayout(preview_container, stretch=1)

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
        # ✅ 新增：开关复选框
        self.dust_checkbox = QCheckBox("启用去灰尘")
        self.dust_checkbox.setChecked(False)
        self.dust_checkbox.stateChanged.connect(self.update_processed_image)

        kernel_size_label = QLabel(
            "形态学核大小：\n（越大清除效果越明显，但也有误清除风险·，建议3-5）"
        )
        self.kernel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_size_slider.setRange(1, 10)
        self.kernel_size_slider.setValue(3)
        self.kernel_size_slider.sliderReleased.connect(self.on_kernel_size_release)
        kernel_size_value_label = QLabel("3")
        kernel_size_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        kernel_size_value_label.setFixedWidth(50)
        self.kernel_size_value_label = kernel_size_value_label

        # 最小连通面积滑块
        min_area_label = QLabel(
            "最小连通面积：\n（面积小于该值的区域将被认为是灰尘清除）"
        )
        self.min_area_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_area_slider.setRange(10, 200)
        self.min_area_slider.setValue(self.min_area)
        self.min_area_slider.sliderReleased.connect(self.on_min_area_release)
        min_area_value_label = QLabel(str(self.min_area))
        min_area_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        min_area_value_label.setFixedWidth(50)
        self.min_area_value_label = min_area_value_label

        dust_inner_layout.addWidget(self.dust_checkbox, 0, 0, 1, 3)  # 跨三列更自然
        # ✅ 滑块从第二行开始
        dust_inner_layout.addWidget(kernel_size_label, 1, 0)
        dust_inner_layout.addWidget(self.kernel_size_slider, 1, 1)
        dust_inner_layout.addWidget(kernel_size_value_label, 1, 2)

        dust_inner_layout.addWidget(min_area_label, 2, 0)
        dust_inner_layout.addWidget(self.min_area_slider, 2, 1)
        dust_inner_layout.addWidget(min_area_value_label, 2, 2)

        dust_group.setLayout(dust_inner_layout)
        dust_layout.addWidget(dust_group)

        params_layout.addLayout(group1)
        params_layout.addLayout(group2)
        params_layout.addLayout(dust_layout)  # 将dust_layout添加到params_layout
        main_layout.addLayout(params_layout)  # 将params_layout添加到main_layout
        # -----------------------------------------
        # ✅ 新增：高斯模糊设置
        gaussian_layout = QVBoxLayout()
        gaussian_group = QGroupBox("高斯模糊设置：\n（让图片边缘更平滑）")
        gaussian_inner_layout = QGridLayout()

        self.gaussian_checkbox = QCheckBox("启用高斯模糊")
        self.gaussian_checkbox.setChecked(False)
        self.gaussian_checkbox.stateChanged.connect(self.update_processed_image)
        # ✅ 新增：高斯模糊参数滑块
        gaussian_kernel_size_label = QLabel("高斯模糊核大小：")
        self.gaussian_kernel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_kernel_size_slider.setRange(3, 15)
        self.gaussian_kernel_size_slider.setSingleStep(2)
        self.gaussian_kernel_size_slider.setPageStep(2)
        self.gaussian_kernel_size_slider.setValue(3)
        self.gaussian_kernel_size_slider.sliderReleased.connect(
            self.on_gaussian_kernel_size_release
        )
        self.gaussian_kernel_size_slider.valueChanged.connect(
            self.on_gaussian_kernel_size_value_change
        )
        gaussian_kernel_size_value_label = QLabel("3")
        gaussian_kernel_size_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        gaussian_kernel_size_value_label.setFixedWidth(50)
        self.gaussian_kernel_size_value_label = gaussian_kernel_size_value_label

        gaussian_inner_layout.addWidget(self.gaussian_checkbox, 0, 0, 1, 3)
        gaussian_inner_layout.addWidget(gaussian_kernel_size_label, 1, 0)
        gaussian_inner_layout.addWidget(self.gaussian_kernel_size_slider, 1, 1)
        gaussian_inner_layout.addWidget(gaussian_kernel_size_value_label, 1, 2)
        gaussian_group.setLayout(gaussian_inner_layout)
        gaussian_layout.addWidget(gaussian_group)
        params_layout.addLayout(gaussian_layout)
        # -----------------------------------------
        # ✅ 新增：锐化处理开关复选框
        # sharpen_layout = QVBoxLayout()
        # sharpen_group = QGroupBox("锐化处理设置：（让图片细节更清晰）")
        # sharpen_inner_layout = QGridLayout()
        # self.sharpen_checkbox = QCheckBox("启用锐化处理")
        # self.sharpen_checkbox.setChecked(False)
        # self.sharpen_checkbox.stateChanged.connect(self.update_processed_image)
        # sharpen_inner_layout.addWidget(self.sharpen_checkbox, 0, 0, 1, 3)
        # sharpen_group.setLayout(sharpen_inner_layout)
        # sharpen_layout.addWidget(sharpen_group)
        # params_layout.addLayout(sharpen_layout)

        # 在init_ui方法的按钮布局部分修改
        btn_layout = QHBoxLayout()
        # 上一张（快捷键P）
        self.prev_btn = QPushButton("上一张（←）(&P)")
        self.prev_btn.clicked.connect(lambda: self.switch_image(-1))
        # 下一张（快捷键N）
        self.next_btn = QPushButton("下一张（→）(&N)")
        self.next_btn.clicked.connect(lambda: self.switch_image(1))
        # 切换模式按钮保持不变
        self.mode_btn = QPushButton("切换模式（当前：白底红字）")
        self.mode_btn.clicked.connect(self.switch_mode)
        # 保存当前图片（快捷键S）
        self.save_current_btn = QPushButton("保存当前图片(&S)")
        self.save_current_btn.clicked.connect(self.save_current)
        # 批量保存所有图片（快捷键B）
        self.save_btn = QPushButton("批量保存所有图片(&B)")
        self.save_btn.clicked.connect(self.batch_save)
        # 退出（快捷键Q）
        self.quit_btn = QPushButton("退出(&Q)")
        self.quit_btn.clicked.connect(QApplication.quit)

        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addWidget(self.mode_btn)
        btn_layout.addWidget(self.save_current_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.quit_btn)
        main_layout.addLayout(btn_layout)
        # 确保窗口获得焦点，以便接收键盘事件
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

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
        self.img_combo.setCurrentIndex(self.img_index)  # ✅ 让下拉框也更新
        print(f"🔄 切换到：{self.file_list[self.img_index]}")

    def switch_mode(self):
        self.output_mode = (self.output_mode + 1) % 3
        mode_names = ["白底红字", "叠加模式", "掩码模式"]
        self.mode_btn.setText(f"切换模式（当前：{mode_names[self.output_mode]}）")
        self.processed_label.setText(f"处理结果（{mode_names[self.output_mode]}）")
        self.update_processed_image()

    def batch_save(self):
        print("📤 开始批量保存...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        for idx, filename in enumerate(self.file_list):
            img_path = os.path.join(self.input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 跳过无法读取的图片：{filename}")
                continue

            # --- 直接调用处理函数 ---
            cleaned = self.process_image(img)

            # --- 保存结果 ---
            name, _ = os.path.splitext(filename)
            save_path = os.path.join(self.output_folder, f"{name}.png")
            cv2.imwrite(save_path, cleaned)
            print(f"✅ 已保存：{save_path}")

        print("🎉 批量保存完成！")
        self.show_toast("批量处理成功!")
        QApplication.restoreOverrideCursor()


if __name__ == "__main__":
    app = QApplication([])
    editor = HSVImageEditor()
    editor.showMaximized()
    app.exec_()
