from datasets import load_dataset
from transformers import AutoTokenizer
import os

def preprocess_function(examples, tokenizer):
    # 提取用户输入和助手回答
    inputs = []
    targets = []

    for conversation in examples["conversations"]:
        user_input = ""
        assistant_response = ""

        # 遍历对话列表
        for turn in conversation:
            if turn["from"] == "user":
                user_input = turn["value"]
            elif turn["from"] == "assistant":
                assistant_response = turn["value"]

        # 如果用户输入或助手回答为空，则跳过该样本
        if not user_input or not assistant_response:
            continue

        # 构造输入和目标文本
        inputs.append(f"问：{user_input}\n答：")
        targets.append(assistant_response)

    # 使用分词器对输入和目标进行编码
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    # 将目标标签添加到模型输入中
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

if __name__ == "__main__":
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    # 加载数据集
    dataset = load_dataset('json', data_files={
        'train': '../data/train_output.json',
        'validation': '../data/val_output.json'
    })

    # 打印数据集结构以确认
    print(dataset)
    print(dataset["train"][0])  # 打印第一条数据

    # 预处理数据
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["id", "conversations"]  # 移除不需要的列
    )

    # 保存预处理后的数据
    os.makedirs("../data/tokenized_datasets", exist_ok=True)
    tokenized_datasets.save_to_disk("../data/tokenized_datasets")