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

file_list = [
    f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
if not file_list:
    raise FileNotFoundError("⚠️ 没有在 input_images 文件夹中找到图片！")

img_index = 0


def load_img(idx):
    img = cv2.imread(os.path.join(input_folder, file_list[idx]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv


img, hsv = load_img(img_index)

compare_mode = "vertical"  # 'horizontal' 或 'vertical'
output_mode = 0  # 0=白底, 1=叠加, 2=掩码


def nothing(x):
    pass


# 创建可缩放窗口（不会模糊）
cv2.namedWindow("Adjust HSV", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Adjust HSV", img.shape[1], img.shape[0])  # 设置窗口大小为图片大小


# Trackbar 参数（和你原来一样）
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

# 设置显示窗口的目标尺寸（根据屏幕调整）
display_width, display_height = 1200, 343  # 你可以改成自己屏幕能显示的尺寸


while True:
    # 获取滑动条的值
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

    # 定义红色范围
    lower_red1 = np.array([lowH1, lowS1, lowV1])
    upper_red1 = np.array([highH1, highS1, highV1])
    lower_red2 = np.array([lowH2, lowS2, lowV2])
    upper_red2 = np.array([highH2, highS2, highV2])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 小膨胀/闭运算，修复缺口
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 三种输出模式
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
        # 模式3: 纯掩码
        cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 显示
    if compare_mode == "horizontal":
        preview = np.hstack((img, cleaned))
    else:
        preview = np.vstack((img, cleaned))

    # 自动缩放显示窗口，保持宽高比
    h, w = preview.shape[:2]
    scale_w = display_width / w
    scale_h = display_height / h
    scale = min(scale_w, scale_h, 1.0)  # scale <= 1，避免放大导致模糊

    new_w, new_h = int(w * scale), int(h * scale)
    preview_resized = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow("Adjust HSV", preview_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        for filename in file_list:
            filepath = os.path.join(input_folder, filename)
            img = cv2.imread(filepath)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

            if output_mode == 0:
                cleaned = np.ones_like(img) * 255
                cleaned[mask > 0] = img[mask > 0]
            elif output_mode == 1:
                background = np.ones_like(img) * 255
                red_only = cv2.bitwise_and(img, img, mask=mask)
                cleaned = cv2.addWeighted(red_only, 1.0, background, 0.0, 0)
            else:
                cleaned = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            savepath = os.path.join(
                output_folder, os.path.splitext(filename)[0] + ".png"
            )
            cv2.imwrite(savepath, cleaned)  # 保存为无损PNG
            print(f"✅ 已保存: {savepath}")
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
