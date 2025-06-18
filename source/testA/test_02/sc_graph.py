import numpy as np
import matplotlib.pyplot as plt
from math import comb
from matplotlib.font_manager import FontProperties

# 指定字体文件路径
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = FontProperties(fname=font_path)

# 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义投票成功概率函数（n=5）
def pvoting(p):
    return sum(comb(5, k) * (p**k) * ((1-p)**(5-k)) for k in range(3, 6))

# 生成 p 的取值范围：从 0 到 1，步长 0.01
p_values = np.linspace(0, 1, 400)
pv_values = [pvoting(p) for p in p_values]

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(p_values, pv_values, label='投票策略准确率 $P_{\\text{voting}}(p)$', color='blue')
plt.plot(p_values, p_values, '--', label='单次准确率 $p$', color='gray')

# 使用指定的中文字体设置标题和标签
plt.title('多数投票策略 vs 单次准确率（n=5）', fontproperties=font_prop)
plt.xlabel('单次生成准确率 $p$', fontproperties=font_prop)
plt.ylabel('最终准确率', fontproperties=font_prop)
plt.grid(True)
plt.legend(prop=font_prop)  # 设置图例字体

plt.tight_layout()

# 保存图像为文件，命名为 sc.png
plt.savefig("sc.png", dpi=300, bbox_inches='tight')

# 显示图像
# plt.show()