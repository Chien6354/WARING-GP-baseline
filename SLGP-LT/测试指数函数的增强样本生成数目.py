# import csv
# import numpy as numpy
#
# def get_feature_requirements(target_count, current_gen, ngen=25, growth_factor=3.0):
#     """
#     针对单个类别的指数增长的增强数量生成函数（返回整数）。
#
#     参数:
#         target_count (int): 该类别最终需要达到的增强数量（gap = max_count - count）
#         current_gen (int): 当前代数（从1到ngen）
#         ngen (int): 总代数（默认25）
#         growth_factor (float): 增长加速因子（默认3.0）
#
#     返回:
#         int: 当前代数下该类别需要增强的样本数量（0..target_count）
#     """
#     target_count = int(target_count)
#     if target_count <= 0:
#         return 0
#
#     # 在第0代或第1代都认为没有增强
#     if current_gen <= 1:
#         return 0
#
#     # 如果只有一代或代数已到达或超过最终代，则直接返回目标值
#     if ngen <= 1 or current_gen >= ngen:
#         return target_count
#
#     # 归一化代数 t 从 0 到 1 （current_gen==1 -> t=0, current_gen==ngen -> t=1）
#     t = (current_gen - 1) / (ngen - 1)
#
#     numerator = numpy.exp(growth_factor * t) - 1.0
#     denominator = numpy.exp(growth_factor) - 1.0
#     if denominator == 0:
#         current_value = target_count * t
#     else:
#         current_value = target_count * (numerator / denominator)
#
#     # 四舍五入并确保不超过 target_count
#     return int(round(min(target_count, current_value)))
#
# def save_feature_requirements_csv(target_counts, ngen=50, growth_factor=3.0, out_file='feature_requirements_ngen50.csv'):
#     """
#     将每代（从0到ngen）每个类别的增强数保存为 CSV。
#     假定 get_feature_requirements(target_count, current_gen, ngen, growth_factor)
#     在当前作用域中可用并按你提供的实现工作。
#
#     参数:
#         target_counts (list[int]): 每个类别的最终目标增强数量，例如 [0,42,...,180]
#         ngen (int): 总代数（默认为50）
#         growth_factor (float): 传入 get_feature_requirements 的增长因子
#         out_file (str): 输出 CSV 文件路径
#
#     CSV 格式:
#         第一列为 gen（代数），后续列为 class_0, class_1, ..., class_9 的当前代增强数（整数）
#     """
#     num_classes = len(target_counts)
#     if num_classes == 0:
#         raise ValueError("target_counts 列表不能为空")
#     # header: gen, class_0, class_1, ...
#     header = ['gen'] + [f'class_{i}' for i in range(num_classes)]
#
#     with open(out_file, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
#         # current_gen 从 0 开始，到 ngen（包含）
#         for gen in range(0, ngen + 1):
#             row_counts = []
#             for tc in target_counts:
#                 need = get_feature_requirements(tc, current_gen=gen, ngen=ngen, growth_factor=growth_factor)
#                 row_counts.append(int(need))
#             writer.writerow([gen] + row_counts)
#
#     return out_file


# 示例用法（按你的要求）
# if __name__ == '__main__':
#     target_counts = [0, 42, 74, 100, 121, 137, 150, 160, 168, 180]
#     out_path = 'feature_requirements_ngen50.csv'
#     save_feature_requirements_csv(target_counts, ngen=50, growth_factor=5, out_file=out_path)
#     print(f"已保存到: {out_path}")

import numpy as np
import matplotlib.pyplot as plt
import math

def get_feature_requirements_continuous(target_count, current_gen, ngen=50, steepness=10.0):
    """返回连续的 sigmoid 值（未取整），用于绘图"""
    if target_count == 0:
        return 0.0
    x = steepness * (current_gen - ngen / 2) / ngen
    sigmoid = 1.0 / (1.0 + math.exp(-x))
    return target_count * sigmoid

# 参数设置
ngen = 50
current_gen_vals = np.linspace(0, ngen, 500)  # 平滑曲线
target_counts = [0, 42, 74, 100, 121, 137, 150, 160, 168, 180]
steepness = 30

# 绘图
plt.figure(figsize=(10, 6))

for tc in target_counts:
    if tc == 0:
        y_vals = np.zeros_like(current_gen_vals)
    else:
        y_vals = [
            get_feature_requirements_continuous(tc, gen, ngen=ngen, steepness=steepness)
            for gen in current_gen_vals
        ]
    plt.plot(current_gen_vals, y_vals, label=f'target={tc}')

# 标注关键点
plt.axvline(x=25, color='gray', linestyle='--', linewidth=0.8, label='Steep rise starts (~gen 25)')

# 图表美化
plt.title('Sigmoid Feature Requirement Curves (ngen=50)')
plt.xlabel('Generation (current_gen)')
plt.ylabel('Feature Requirement (y)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 50)
plt.ylim(0, max(target_counts) * 1.05)
plt.tight_layout()
plt.show()