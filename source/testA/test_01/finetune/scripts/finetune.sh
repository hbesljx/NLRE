#!/bin/bash

# 激活虚拟环境（如果需要）
# source venv/bin/activate

# 切换到项目根目录
cd ..

# 运行微调脚本
echo "Starting fine-tuning..."
python src/finetune.py

# 如果使用 DeepSpeed，可以替换为以下命令：
# deepspeed src/finetune.py

echo "Fine-tuning completed!"