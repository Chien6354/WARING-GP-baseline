import os
import glob

# 根目录
root_dir = r"D:\Code\longtail_multi_fewshot_GP\MEGPE\消融实验单目标\MEGPE_contrast\pkl"

# 遍历一级文件夹
for first_level in os.listdir(root_dir):
    first_path = os.path.join(root_dir, first_level)

    # 确保是一级文件夹（不是文件）
    if not os.path.isdir(first_path):
        continue

    print(f"进入一级文件夹: {first_level}")

    # 遍历二级文件夹
    for second_level in os.listdir(first_path):
        second_path = os.path.join(first_path, second_level)

        # 确保是二级文件夹
        if not os.path.isdir(second_path):
            continue

        # 匹配该二级文件夹下所有 .log 文件
        log_files = glob.glob(os.path.join(second_path, "*.log"))

        print(f"  处理二级文件夹: {second_level}，发现 {len(log_files)} 个 .log 文件")

        # 删除每个 .log 文件
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"    已删除: {os.path.basename(log_file)}")
            except Exception as e:
                print(f"    删除失败: {log_file}, 错误: {e}")