import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 读取CSV文件
try:
    # 尝试用制表符分隔读取
    df = pd.read_csv('humen.csv', sep=',')
    
    # 检查是否有"可读性 综合评价"这样的列（由于格式问题）
    problematic_columns = [col for col in df.columns if ' ' in col]
    for col in problematic_columns:
        if '可读性' in col:
            # 分割该列为两列
            parts = col.split(' ')
            df['可读性'] = df[col].apply(lambda x: str(x).split(' ')[0])
            df = df.drop(columns=[col])
            break
            
except Exception as e:
    print(f"读取文件出错: {e}")
    raise

# 将所有列转换为数值类型
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 提取五列的值
comprehensiveness = df['全面性'].tolist()
consistency = df['一致性'].tolist()
executability = df['可执行性'].tolist()
readability = df['可读性'].tolist()

# 将所有数据放入列表中
all_data = [comprehensiveness, consistency, executability, readability]
labels = ['Comprehensiveness', 'Consistency', 'Executability', 'Readability']

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 自定义颜色列表
custom_palette = ["#66c2a5", "#a6d854", "#8da0cb", "#e78ac3"]

# 绘制小提琴图，确保所有小提琴都有中轴线
sns.violinplot(data=all_data, ax=ax, palette=custom_palette, cut=0.07)
ax.set_title("Score Distribution", fontsize=18)

# 手动计算并绘制每个数据组的均值和中位数
for i, data in enumerate(all_data):
    data = [d for d in data if not np.isnan(d)]  # 过滤掉NaN值
    if data:
        mean = sum(data) / len(data)
        ax.hlines(mean, i - 0.022, i + 0.022, color='#a0cbe8', linewidth=1.25)
        
        sorted_data = sorted(data)
        median = sorted_data[len(sorted_data) // 2]
        ax.hlines(median, i - 0.022, i + 0.022, color='#f4a6a1', linewidth=1.25)

# 设置x轴标签
ax.set_xticks(range(len(all_data)))
ax.set_xticklabels(labels, fontsize=14)

# 设置y轴标签
ax.set_ylabel("Score", fontsize=16)

# 减少y轴刻度密度
# 计算所有非NaN值的最小和最大值
# all_values = []
# for data in all_data:
#     all_values.extend([d for d in data if not np.isnan(d)])

# if all_values:
#     min_score = min(all_values)
#     max_score = max(all_values)
    
#     # 设置y轴刻度为2的倍数（减少刻度密度）
#     yticks = np.arange(int(np.floor(min_score)), int(np.ceil(max_score)) + 1, 1)
#     ax.set_yticks(yticks)

# 设置x轴和y轴刻度的字体大小
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=14)

# 添加图例并放在坐标轴外
median_patch = plt.Line2D([0], [0], color="#f4a6a1", lw=4, label='Median')
mean_patch = plt.Line2D([0], [0], color="#a0cbe8", lw=4, label='Mean')
ax.legend(handles=[median_patch, mean_patch], loc='center left', bbox_to_anchor=(1, 0.94), fontsize=15)

# 保存图形为PDF文件
plt.savefig("violin_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig("violin_plot.png", format="png", dpi=300, bbox_inches="tight")
