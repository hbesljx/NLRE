import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 根据logs目录下的日志文件生成loss和learning_rate曲线图

# 指定日志目录
log_dir = "./logs"  # 替换为你的日志目录

# 获取所有事件文件
event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]

# 初始化画布
plt.figure(figsize=(10, 6))

# 遍历每个事件文件
for i, event_file in enumerate(event_files):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()  # 加载数据

    # 获取所有标签
    tags = ea.Tags()
    print(f"Processing file: {event_file}")
    print(f"Available tags: {tags}")

    # 提取训练损失数据
    if 'train/loss' in tags['scalars']:
        loss_events = ea.Scalars('train/loss')
        if loss_events:  # 检查是否为空
            steps = [e.step for e in loss_events]
            values = [e.value for e in loss_events]

            # 绘制折线图
            plt.plot(steps, values, label=f'Train Loss (File {i + 1})')

    # 提取验证损失数据
    if 'eval/loss' in tags['scalars']:
        eval_loss_events = ea.Scalars('eval/loss')
        if eval_loss_events:  # 检查是否为空
            eval_steps = [e.step for e in eval_loss_events]
            eval_values = [e.value for e in eval_loss_events]

            # 绘制折线图
            plt.plot(eval_steps, eval_values, label=f'Eval Loss (File {i + 1})', linestyle='--')

# 设置图表标题和标签
plt.title('Training and Evaluation Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# 显示图表
# plt.show()
# 显示图表改为保存图表
plt.savefig('training_evaluation_loss.png')