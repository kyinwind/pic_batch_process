import cv2
import numpy as np
import os

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

# 输入输出文件夹
input_folder = "input_images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# 筛选图片文件（增加对大小写扩展名的兼容）
file_list = [
    f
    for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
]
if not file_list:
    raise FileNotFoundError("⚠️ 没有在 input_images 文件夹中找到图片！")

img_index = 0


def load_img(idx):
    """加载图片，增加错误处理，避免空图像崩溃"""
    img_path = os.path.join(input_folder, file_list[idx])
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ 无法读取图片：{img_path}（检查路径或文件完整性）")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv


# 初始加载第一张图
img, hsv = load_img(img_index)

# 配置参数（优化默认值）
compare_mode = "vertical"  # 'horizontal'（左右）或 'vertical'（上下）
output_mode = 0  # 0=白底, 1=叠加, 2=掩码
MAX_PREVIEW_SIZE = (1200, 800)  # 预览窗口最大尺寸（宽，高），适配大多数屏幕


def nothing(x):
    pass


# 1. 创建可缩放窗口（不提前固定大小，避免拉伸）
cv2.namedWindow("Adjust HSV", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# WINDOW_KEEPRATIO：保证窗口缩放时，图像保持原始宽高比


# 2. 创建Trackbar（保持原逻辑）
cv2.createTrackbar("LowH1", "Adjust HSV", 0, 180, nothing)
cv2.createTrackbar("HighH1", "Adjust HSV", 10, 180, nothing)
cv2.createTrackbar("LowS1", "Adjust HSV", 80, 255, nothing)
cv2.createTrackbar("HighS1", "Adjust HSV", 255, 255, nothing)
cv2.createTrackbar("LowV1", "Adjust HSV", 80, 255, nothing)
cv2.createTrackbar("HighV1", "Adjust HSV", 255, 255, nothing)

cv2.createTrackbar("LowH2", "Adjust HSV", 170, 180, nothing)
cv2.createTrackbar("HighH2", "Adjust HSV", 180, 180, nothing)
cv2.createTrackbar("LowS2", "Adjust HSV", 80, 255, nothing)
cv2.createTrackbar("HighS2", "Adjust HSV", 255, 255, nothing)
cv2.createTrackbar("LowV2", "Adjust HSV", 80, 255, nothing)
cv2.createTrackbar("HighV2", "Adjust HSV", 255, 255, nothing)


def calculate_optimal_scale(original_size, max_size):
    """
    计算最优缩放比例：保证图像缩放后不超过最大尺寸，且保持宽高比
    original_size: (原宽, 原高)
    max_size: (最大宽, 最大高)
    """
    orig_w, orig_h = original_size
    max_w, max_h = max_size

    # 计算宽度和高度的缩放比例（取较小值，避免超出最大尺寸）
    scale_w = max_w / orig_w if orig_w != 0 else 1.0
    scale_h = max_h / orig_h if orig_h != 0 else 1.0
    scale = min(scale_w, scale_h, 1.0)  # 不放大（scale≤1），避免放大导致模糊
    return scale


while True:
    # 3. 获取Trackbar值（保持原逻辑）
    lowH1 = cv2.getTrackbarPos("LowH1", "Adjust HSV")
    highH1 = cv2.getTrackbarPos("HighH1", "Adjust HSV")
    lowS1 = cv2.getTrackbarPos("LowS1", "Adjust HSV")
    highS1 = cv2.getTrackbarPos("HighS1", "Adjust HSV")
    lowV1 = cv2.getTrackbarPos("LowV1", "Adjust HSV")
    highV1 = cv2.getTrackbarPos("HighV1", "Adjust HSV")

    lowH2 = cv2.getTrackbarPos("LowH2", "Adjust HSV")
    highH2 = cv2.getTrackbarPos("HighH2", "Adjust HSV")
    lowS2 = cv2.getTrackbarPos("LowS2", "Adjust HSV")
    highS2 = cv2.getTrackbarPos("HighS2", "Adjust HSV")
    lowV2 = cv2.getTrackbarPos("LowV2", "Adjust HSV")
    highV2 = cv2.getTrackbarPos("HighV2", "Adjust HSV")

    # 4. 红色掩码计算（保持原逻辑）
    lower_red1 = np.array([lowH1, lowS1, lowV1])
    upper_red1 = np.array([highH1, highS1, highV1])
    lower_red2 = np.array([lowH2, lowS2, lowV2])
    upper_red2 = np.array([highH2, highS2, highV2])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 小膨胀/闭运算，修复缺口（保持原逻辑）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 5. 三种输出模式（保持原逻辑）
    if output_mode == 0:
        # 模式1: 白底红字
        cleaned = np.ones_like(img) * 255
        cleaned[mask > 0] = img[mask > 0]
    elif output_mode == 1:
        # 模式2: 原图叠加（只保留红字，背景淡化）
        background = np.ones_like(img) * 255
        red_only = cv2.bitwise_and(img, img, mask=mask)
        cleaned = cv2.addWeighted(red_only, 1.0, background, 0.0, 0)
    else:
        # 模式3: 纯掩码（转为BGR以便和原图拼接）
        cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 6. 拼接原图和处理图（优化缩放逻辑）
    if compare_mode == "horizontal":
        # 左右拼接：宽度相加，高度取两者最大（避免截断）
        combined_h = max(img.shape[0], cleaned.shape[0])
        # 统一高度（用INTER_AREA保持细节）
        img_resized = cv2.resize(
            img, (img.shape[1], combined_h), interpolation=cv2.INTER_AREA
        )
        cleaned_resized = cv2.resize(
            cleaned, (cleaned.shape[1], combined_h), interpolation=cv2.INTER_AREA
        )
        preview = np.hstack((img_resized, cleaned_resized))
    else:
        # 上下拼接：高度相加，宽度取两者最大
        combined_w = max(img.shape[1], cleaned.shape[1])
        # 统一宽度（用INTER_AREA保持细节）
        img_resized = cv2.resize(
            img, (combined_w, img.shape[0]), interpolation=cv2.INTER_AREA
        )
        cleaned_resized = cv2.resize(
            cleaned, (combined_w, cleaned.shape[0]), interpolation=cv2.INTER_AREA
        )
        preview = np.vstack((img_resized, cleaned_resized))

    # 7. 计算最优缩放比例，避免模糊
    preview_orig_size = (preview.shape[1], preview.shape[0])  # (宽, 高)
    scale = calculate_optimal_scale(preview_orig_size, MAX_PREVIEW_SIZE)
    # 缩放预览图（关键：用INTER_AREA插值，缩小图像时细节保留最好）
    preview_scaled = cv2.resize(
        preview,
        (int(preview_orig_size[0] * scale), int(preview_orig_size[1] * scale)),
        interpolation=cv2.INTER_AREA,  # 替换为INTER_AREA，解决缩小模糊
    )

    # 8. 显示缩放后的图像
    cv2.imshow("Adjust HSV", preview_scaled)
    # 自动调整窗口大小以匹配缩放后的图像（避免黑边或拉伸）
    cv2.resizeWindow("Adjust HSV", preview_scaled.shape[1], preview_scaled.shape[0])

    # 9. 键盘控制（保持原逻辑）
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # 批量处理并保存（优化保存逻辑：保留原扩展名，避免强制转PNG）
        for filename in file_list:
            filepath = os.path.join(input_folder, filename)
            img_save = cv2.imread(filepath)
            if img_save is None:
                print(f"⚠️ 跳过无法读取的图片：{filepath}")
                continue
            hsv_save = cv2.cvtColor(img_save, cv2.COLOR_BGR2HSV)

            # 重新计算掩码（避免复用之前的mask，确保每张图独立处理）
            mask1_save = cv2.inRange(hsv_save, lower_red1, upper_red1)
            mask2_save = cv2.inRange(hsv_save, lower_red2, upper_red2)
            mask_save = cv2.bitwise_or(mask1_save, mask2_save)
            mask_save = cv2.morphologyEx(mask_save, cv2.MORPH_CLOSE, kernel)
            mask_save = cv2.dilate(mask_save, kernel, iterations=1)

            # 生成处理后的图像
            if output_mode == 0:
                cleaned_save = np.ones_like(img_save) * 255
                cleaned_save[mask_save > 0] = img_save[mask_save > 0]
            elif output_mode == 1:
                background_save = np.ones_like(img_save) * 255
                red_only_save = cv2.bitwise_and(img_save, img_save, mask=mask_save)
                cleaned_save = cv2.addWeighted(
                    red_only_save, 1.0, background_save, 0.0, 0
                )
            else:
                cleaned_save = cv2.cvtColor(mask_save, cv2.COLOR_GRAY2BGR)

            # 无论原文件是JPG/BMP，均保存为“原文件名.png”
            name, _ = os.path.splitext(filename)  # 忽略原扩展名
            savepath = os.path.join(output_folder, f"{name}.png")
            # PNG为无损格式，无需额外设置质量参数，直接保存
            cv2.imwrite(savepath, cleaned_save)
            print(f"✅ 已保存为PNG: {savepath}")

        print("🎉 所有图片处理完成！")
        break
    elif key == ord("q"):
        print("❌ 用户退出。")
        break
    elif key == ord("c"):
        compare_mode = "vertical" if compare_mode == "horizontal" else "horizontal"
        print(f"🔄 已切换为 {'上下比较' if compare_mode == 'vertical' else '左右比较'}")
    elif key == ord("m"):
        output_mode = (output_mode + 1) % 3
        print(
            f"🎨 已切换输出模式: {output_mode} ({['白底','叠加','掩码'][output_mode]})"
        )
    elif key == ord("p"):
        img_index = (img_index - 1) % len(file_list)
        img, hsv = load_img(img_index)
        print(f"⬆️ 切换到上一张：{file_list[img_index]}")
    elif key == ord("n"):
        img_index = (img_index + 1) % len(file_list)
        img, hsv = load_img(img_index)
        print(f"⬇️ 切换到下一张：{file_list[img_index]}")

cv2.destroyAllWindows()
