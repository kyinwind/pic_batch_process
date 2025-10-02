import os


def remove_copy_suffix(folder_path):
    # 获取文件夹中所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以"拷贝"结尾
        print(filename)
        if filename.endswith(" 拷贝"):
            # 构建旧文件路径和新文件路径
            old_path = os.path.join(folder_path, filename)
            new_filename = filename[:-2]  # 移除末尾两个字符（"拷贝"）
            new_path = os.path.join(folder_path, new_filename)

            # 避免文件名重复
            counter = 1
            while os.path.exists(new_path):
                # 如果新文件名已存在，添加数字后缀
                name, ext = os.path.splitext(new_filename)
                new_path = os.path.join(folder_path, f"{name}_{counter}{ext}")
                counter += 1

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"已重命名: {filename} -> {os.path.basename(new_path)}")
        elif "." in filename:
            # 处理带扩展名的文件，如"文件拷贝.jpg"
            name, ext = os.path.splitext(filename)
            if name.endswith("拷贝"):
                old_path = os.path.join(folder_path, filename)
                new_name = name[:-2] + ext  # 移除名称部分的"拷贝"
                new_path = os.path.join(folder_path, new_name)

                # 避免文件名重复
                counter = 1
                while os.path.exists(new_path):
                    name_part, ext_part = os.path.splitext(new_name)
                    new_path = os.path.join(
                        folder_path, f"{name_part}_{counter}{ext_part}"
                    )
                    counter += 1

                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {os.path.basename(new_path)}")


if __name__ == "__main__":
    # 替换为你的文件夹路径
    target_folder = "C:/dev/pic_batch_process/input_images"

    # 验证文件夹是否存在
    if not os.path.isdir(target_folder):
        print(f"错误: 文件夹 '{target_folder}' 不存在")
    else:
        remove_copy_suffix(target_folder)
        print("批量重命名完成")
