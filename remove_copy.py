import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# 支持的图片文件扩展名
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")


def is_image_file(filename):
    """判断文件是否为图片文件"""
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def process_images(folder_path):
    """处理指定目录下的图片文件"""
    if not os.path.isdir(folder_path):
        messagebox.showerror("错误", f"文件夹 '{folder_path}' 不存在")
        return

    processed_count = 0

    # 获取文件夹中所有文件
    for filename in os.listdir(folder_path):
        # 只处理图片文件
        if not is_image_file(filename):
            continue

        # 保存原始文件名用于输出信息
        original_filename = filename

        # 检查文件名是否以" 拷贝"结尾（无扩展名的情况）
        if filename.endswith(" 拷贝"):
            # 移除末尾的" 拷贝"（包含前面的空格）
            filename = filename[:-3]
        else:
            # 处理带扩展名的文件
            name, ext = os.path.splitext(filename)
            if name.endswith("拷贝"):
                # 移除名称部分的"拷贝"
                filename = name[:-2] + ext

        # 去除文件名中的所有空格
        filename = filename.replace(" ", "")

        # 如果文件名有变化才进行重命名
        if filename != original_filename:
            # 构建旧文件路径和新文件路径
            old_path = os.path.join(folder_path, original_filename)
            new_path = os.path.join(folder_path, filename)

            # 避免文件名重复
            counter = 1
            while os.path.exists(new_path):
                # 如果新文件名已存在，添加数字后缀
                name_part, ext_part = os.path.splitext(filename)
                new_path = os.path.join(folder_path, f"{name_part}_{counter}{ext_part}")
                counter += 1

            # 执行重命名
            os.rename(old_path, new_path)
            print(f"已重命名: {original_filename} -> {os.path.basename(new_path)}")
            processed_count += 1

    messagebox.showinfo("完成", f"批量处理完成，共处理了 {processed_count} 个图片文件")


def select_folder():
    """让用户选择文件夹并处理其中的图片"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    folder_path = filedialog.askdirectory(title="选择图片所在的文件夹")

    if folder_path:  # 如果用户选择了文件夹而不是取消
        process_images(folder_path)


if __name__ == "__main__":
    select_folder()
