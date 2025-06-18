import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 强制使用 GPU 0 和 GPU 1
torch.cuda.set_device(0)  # 设置主设备为 GPU 0

# 定义量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用 4-bit 量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载模型和分词器
model_dir = './output/final_model'  # 微调后保存的模型路径
base_model_name = "Qwen/Qwen2.5-14B-Instruct"  # 基础模型名称

# 加载基础模型并应用量化
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# 加载 LoRA 微调后的增量参数
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()

# 加载分词器，并设置左填充
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")

# 定义文件路径
input_file = './data/test_data.xlsx'  # 输入的 Excel 文件路径
output_file = './data/output_with_chain_of_thought.xlsx'  # 输出的 Excel 文件路径

# 读取 Excel 文件
df = pd.read_excel(input_file)

# 确保有三列：A列（描述）、B列（正则表达式）、C列（思维链）
if '描述' not in df.columns:
    raise ValueError("Excel 文件中缺少 '描述' 列")
if '正则表达式' not in df.columns:
    raise ValueError("Excel 文件中缺少 '正则表达式' 列")
if '思维链' not in df.columns:
    df['思维链'] = None  # 如果没有 '思维链' 列，则创建一个空列
if '最终正则表达式' not in df.columns:
    df['最终正则表达式'] = None  # 如果没有 '最终正则表达式' 列，则创建一个空列

# 定义一个函数来生成思维链
def generate_chain_of_thought(prompt):
    try:
        # 按照微调时的输入格式构造完整的对话文本
        full_text = f"User: {prompt}\nAssistant:"

        # 将输入文本转换为模型可接受的格式
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to("cuda")

        # 使用模型生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 最大生成 512 个新 token
                num_beams=5,         # 使用 beam search
                do_sample=True       # 不使用贪婪解码
            )

        # 解码生成的输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 "Assistant:" 后的内容
        assistant_start = response.find("Assistant:")
        if assistant_start != -1:
            response = response[assistant_start + len("Assistant:"):].strip()
        else:
            response = response.strip()

        # 返回生成的内容，不做任何修改
        return response
    except Exception as e:
        print(f"Error generating chain of thought for prompt '{prompt}': {e}")
        return "生成失败"

# 遍历 DataFrame 并生成思维链
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    description = row['描述']
    if pd.isna(description):  # 跳过空值
        continue

    # 生成思维链
    chain_of_thought = generate_chain_of_thought(description)

    # 更新 DataFrame
    df.at[idx, '思维链'] = chain_of_thought

    # 验证正则表达式是否被修改
    original_regex = row['正则表达式']
    if pd.notna(original_regex) and original_regex in chain_of_thought:
        print(f"第 {idx + 1} 行的正则表达式未被修改。")
    else:
        print(f"警告：第 {idx + 1} 行的正则表达式可能被修改，请检查！")

# 保存结果到新的 Excel 文件
df.to_excel(output_file, index=False)
print(f"处理完成，结果已保存到 {output_file}")