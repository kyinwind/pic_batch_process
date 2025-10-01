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

# 读取第一张图片（用于调节参数）
file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not file_list:
    raise FileNotFoundError("⚠️ 没有在 input_images 文件夹中找到图片！")
img = cv2.imread(os.path.join(input_folder, file_list[0]))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

compare_mode = 'horizontal'  # 'horizontal'（左右）或'vertical'（上下）


# 回调函数（空的即可）
def nothing(x):
    pass

# 创建窗口和滑动条
cv2.namedWindow("Adjust HSV")

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

    # 定义两个红色范围
    lower_red1 = np.array([lowH1, lowS1, lowV1])
    upper_red1 = np.array([highH1, highS1, highV1])
    lower_red2 = np.array([lowH2, lowS2, lowV2])
    upper_red2 = np.array([highH2, highS2, highV2])

    # 生成掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 生成纯白背景并保留红字
    cleaned = np.ones_like(img) * 255
    cleaned[mask > 0] = img[mask > 0]

    # 显示预览
    if compare_mode == 'horizontal':
        preview = np.hstack((img, cleaned))
    else:
        preview = np.vstack((img, cleaned))
    cv2.imshow("Adjust HSV", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  
        # 保存所有图片
        for filename in file_list:
            filepath = os.path.join(input_folder, filename)
            img = cv2.imread(filepath)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            cleaned = np.ones_like(img) * 255
            cleaned[mask > 0] = img[mask > 0]

            savepath = os.path.join(output_folder, filename)
            #cv2.imwrite(savepath, cleaned)
            cv2.imwrite(savepath, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"✅ 已保存: {filename}")
        print("🎉 所有图片处理完成！")
        break
    elif key == ord('q'):  
        print("❌ 用户退出。")
        break
    elif key == ord('c'):
        # 切换比较模式
        compare_mode = 'vertical' if compare_mode == 'horizontal' else 'horizontal'
        print(f"🔄 已切换为 {'上下比较' if compare_mode == 'vertical' else '左右比较'}")

cv2.destroyAllWindows()
