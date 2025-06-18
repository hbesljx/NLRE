import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

# 指定字体文件路径
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = FontProperties(fname=font_path)

# 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义实验标签（只保留 00, 01, 02）
experiments = ["00_Initial", "01_Finetuned_NoSC", "02_Finetuned_SC_Robust"]

# 自定义显示标签（中文名称映射）
display_labels = {
    "Initial": "初始大模型",
    "Finetuned_NoSC": "微调后大模型（无Self-Consistency）",
    "Finetuned_SC_Robust": "微调后大模型（有Self-Consistency）"
}

# 提取原始标签名（用于匹配 experiments 字典）
labels = [exp.split("_", 1)[1] for exp in experiments]

# 将原始标签转换为中文显示标签
zh_labels = [display_labels.get(label, label) for label in labels]

# 加载 CSV 文件并提取等价比例和部分匹配率
def load_results(experiment_id):
    file_path = f"./part_logs/regex_comparison_results_{experiment_id}.csv"
    print(f"尝试加载文件: {file_path}")
    try:
        df = pd.read_csv(file_path)
        equivalent_ratio = df["IsEquivalent"].mean()  # 等价率
        partial_match_ratio = (df["SimilarityScore"] >= 0.8).mean()  # 部分匹配率
        return equivalent_ratio, partial_match_ratio
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None, None

# 加载所有实验的结果
results_equivalent = {}
results_partial_match = {}
for exp in experiments:
    experiment_id = exp[:2]  # 提取前两位作为实验 ID (如 00, 01, 02)
    equivalent_ratio, partial_match_ratio = load_results(experiment_id)
    if equivalent_ratio is not None and partial_match_ratio is not None:
        results_equivalent[exp] = equivalent_ratio
        results_partial_match[exp] = partial_match_ratio
    else:
        print(f"实验 {exp} 的结果为空，跳过。")

# 提取数据
labels = [exp.split("_", 1)[1] for exp in experiments]  # 如 Initial, Finetuned_NoSC...
equivalent_ratios = [results_equivalent.get(exp, 0) for exp in experiments]
partial_match_ratios = [results_partial_match.get(exp, 0) for exp in experiments]

print("实验标签:", labels)
print("等价比例:", equivalent_ratios)
print("部分匹配比例:", partial_match_ratios)

# 绘制条形图和两条折线图（只画一个图）
def plot_bar_and_two_lines_chart(labels, equivalent_ratios, partial_match_ratios, selected_labels, title, save_path):
    # 筛选需要的实验
    selected_equiv_ratios = []
    selected_partial_ratios = []
    for label in selected_labels:
        if label in labels:
            idx = labels.index(label)
            selected_equiv_ratios.append(equivalent_ratios[idx])
            selected_partial_ratios.append(partial_match_ratios[idx])
        else:
            print(f"警告: 标签 '{label}' 未找到，跳过该实验。")
    
    if len(selected_equiv_ratios) == 0 or len(selected_partial_ratios) == 0:
        print("没有有效的实验数据，跳过绘图。")
        return

    print(f"筛选后的标签: {selected_labels}")
    print(f"筛选后的等价比例: {selected_equiv_ratios}")
    print(f"筛选后的部分匹配比例: {selected_partial_ratios}")

    # 绘图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(selected_labels))  # x轴位置
    bar_width = 0.35  # 条形宽度

    # 条形图：等价率
    plt.bar(x - bar_width/2, selected_equiv_ratios, width=bar_width, color='blue', alpha=0.7, label="等价率 (Bar)")

    # 条形图：部分匹配率
    plt.bar(x + bar_width/2, selected_partial_ratios, width=bar_width, color='green', alpha=0.7, label="部分匹配率 (Bar)")

    # 折线图：等价率
    plt.plot(x - bar_width/2, selected_equiv_ratios, color='red', marker='o', linestyle='-', linewidth=2, label="等价率 (Line)")

    # 折线图：部分匹配率
    plt.plot(x + bar_width/2, selected_partial_ratios, color='purple', marker='s', linestyle='--', linewidth=2, label="部分匹配率 (Line)")

    # 添加数值标签
    for i, value in enumerate(selected_equiv_ratios):
        plt.text(i - bar_width/2, value + 0.01, f"{value:.2%}", ha="center", fontsize=10, fontproperties=font_prop)
    for i, value in enumerate(selected_partial_ratios):
        plt.text(i + bar_width/2, value + 0.01, f"{value:.2%}", ha="center", fontsize=10, fontproperties=font_prop)

    # 手动设置字体
    plt.xticks(x, selected_labels, fontproperties=font_prop)
    plt.ylabel("比例", fontproperties=font_prop)
    plt.title(title, fontproperties=font_prop)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(prop=font_prop)
    plt.tight_layout()
    plt.savefig(save_path)  # 保存图片
    plt.show()

if __name__ == "__main__":
    os.makedirs("./part_plots", exist_ok=True)

    # 使用中文标签绘制第一个图表：00, 01, 02
    plot_bar_and_two_lines_chart(
        zh_labels,  # 这里使用中文标签
        equivalent_ratios,
        partial_match_ratios,
        selected_labels=[
            "初始大模型",
            "微调后大模型（无Self-Consistency）",
            "微调后大模型（有Self-Consistency）"
        ],
        title="鲁棒数据集下的等价率与部分匹配率对比",
        save_path="./part_plots/bar_two_lines_chart_00_01_02.png"
    )