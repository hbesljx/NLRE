import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体（请确认路径正确）
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # 文泉驿正黑字体
font_prop = FontProperties(fname=font_path, size=12)  # 指定字体大小

# 确保保存目录存在
os.makedirs("./memory_plots", exist_ok=True)

# 实验标签与对应的Excel文件名
experiments = {
    "初始大模型": "resource_usage_00.xlsx",
    "微调后大模型（无Self-Consistency）": "resource_usage_01.xlsx",
    "微调后大模型（有Self-Consistency）": "resource_usage_02.xlsx"
}

# 加载数据
data = {}
for label, filename in experiments.items():
    file_path = os.path.join("./memory_logs", filename)
    if os.path.exists(file_path):
        data[label] = pd.read_excel(file_path)
    else:
        print(f"警告：文件 {file_path} 不存在")

if len(data) < len(experiments):
    raise ValueError("部分资源文件未找到，请检查路径。")

# 绘图函数，统一使用 font_prop
def plot_metric(df_dict, metric_name, y_label, title, save_filename):
    plt.figure(figsize=(14, 7))
    for label, df in df_dict.items():
        plt.plot(df.index, df[metric_name], label=label, linewidth=1.5)
    plt.title(title, fontproperties=font_prop, fontsize=14)
    plt.xlabel('样本编号', fontproperties=font_prop, fontsize=12)
    plt.ylabel(y_label, fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("./memory_plots", save_filename), dpi=300, bbox_inches='tight')
    plt.close()

# 依次绘制每个指标
plot_metric(data, '处理时间', '处理时间 (秒)', '不同实验条件下的处理时间对比', 'processing_time_comparison.png')
plot_metric(data, 'CPU使用率', 'CPU使用率 (%)', '不同实验条件下的CPU使用率对比', 'cpu_usage_comparison.png')
plot_metric(data, '内存使用RSS', '内存使用RSS (MB)', '不同实验条件下的内存使用RSS对比', 'memory_rss_comparison.png')
plot_metric(data, '内存使用VMS', '内存使用VMS (MB)', '不同实验条件下的内存使用VMS对比', 'memory_vms_comparison.png')

print("✅ 图表生成完成，已保存至 ./memory_plots 目录")