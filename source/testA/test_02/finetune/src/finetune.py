import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# 数据加载函数
def load_data(train_path, val_path):
    # 加载 JSON 数据
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    # 定义预处理函数
    def preprocess_function(examples):
        inputs = []
        labels = []

        for conversation in examples["conversations"]:
            # 拼接用户输入和助手输出
            user_input = conversation[0]["value"]
            assistant_output = conversation[1]["value"]

            # 构造完整的对话文本
            full_text = f"User: {user_input}\nAssistant: {assistant_output}"

            # 分词并生成 input_ids 和 labels
            tokenized = tokenizer(
                full_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            inputs.append(tokenized["input_ids"].squeeze(0))
            labels.append(tokenized["input_ids"].squeeze(0))  # labels 与 input_ids 相同

        return {"input_ids": inputs, "labels": labels}

    # 对数据集进行预处理
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["id", "conversations"],  # 移除不需要的列
        num_proc=4  # 使用 4 个线程并行处理
    )

    return tokenized_datasets, tokenizer


# 主函数
def main():
    # 数据路径
    train_path = "/root/ljx/nlre/chat/Qwen/test/test_02/finetune/data/train_output.json"
    val_path = "/root/ljx/nlre/chat/Qwen/test/test_02/finetune/data/val_output.json"

    # 加载和预处理数据
    tokenized_datasets, tokenizer = load_data(train_path, val_path)

    # 定义量化配置 (BitsAndBytes)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 使用 4-bit 量化
        bnb_4bit_use_double_quant=True,  # 双重量化
        bnb_4bit_quant_type="nf4",  # 使用 NF4 量化类型
        bnb_4bit_compute_dtype="float16"  # 计算时使用 FP16
    )

    # 加载模型并应用量化
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct",
        quantization_config=quantization_config,
        device_map="auto"  # 自动分配到可用设备
    )

    # 准备模型以支持 k-bit 训练
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 配置 LoRA
    lora_config = LoraConfig(
        r=8,  # 低秩分解的秩
        lora_alpha=32,  # 缩放因子
        target_modules=["q_proj", "v_proj"],  # 需要应用 LoRA 的模块
        lora_dropout=0.1,  # Dropout 率
        bias="none",  # 不更新偏置
        task_type="SEQ_2_SEQ_LM"  # 任务类型为序列到序列建模
    )

    # 应用 LoRA
    model = get_peft_model(model, lora_config)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="../output/results",  # 输出目录
        per_device_train_batch_size=8,  # 每个设备的训练批次大小
        per_device_eval_batch_size=8,  # 每个设备的验证批次大小
        gradient_accumulation_steps=1,  # 梯度累积步数
        num_train_epochs=3,  # 训练轮数
        logging_dir="../output/logs",  # 日志目录
        logging_steps=1,  # 每 1 步记录一次日志
        save_steps=1000,  # 每 1000 步保存一次模型
        evaluation_strategy="steps",  # 验证策略
        eval_steps=1000,  # 每 1000 步验证一次
        save_total_limit=2,  # 最多保存 2 个模型
        fp16=False,  # 关闭 FP16
        bf16=True,  # 启用 BF16（适用于 RTX 4090）
        deepspeed="./ds_config.json" if os.path.exists("./ds_config.json") else None  # 如果使用 DeepSpeed
    )

    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # 训练数据集
        eval_dataset=tokenized_datasets["validation"],  # 验证数据集
        tokenizer=tokenizer
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model("../output/final_model")


if __name__ == "__main__":
    main()