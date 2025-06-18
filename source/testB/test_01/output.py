import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import psutil

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
output_file = './data/output_with_chain_of_thought.xlsx'  # 包含原始数据和思维链的输出文件路径
resource_usage_file = './data/resource_usage.xlsx'  # 资源使用情况和时间信息的输出文件路径

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

# 新增函数：获取当前进程资源使用情况
def get_resource_usage():
    process = psutil.Process()
    cpu_percent = process.cpu_percent(interval=0.1) / psutil.cpu_count()  # 单位 %
    memory_info = process.memory_info()
    return {
        'cpu_percent': cpu_percent,
        'memory_rss': memory_info.rss / (1024 * 1024),  # 转换为 MB
        'memory_vms': memory_info.vms / (1024 * 1024)   # 转换为 MB
    }

# 创建新的 DataFrame 用于记录资源使用情况和时间信息
resource_df = pd.DataFrame(columns=['描述', '处理时间', 'CPU使用率', '内存使用RSS', '内存使用VMS'])

# 修改 generate_chain_of_thought 函数以包括时间和资源统计
def generate_chain_of_thought(prompt):
    start_time = time.time()
    resource_start = get_resource_usage()

    try:
        full_text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 最大生成 512 个新 token
                num_beams=5,         # 使用 beam search
                do_sample=True       # 不使用贪婪解码
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_start = response.find("Assistant:")
        if assistant_start != -1:
            response = response[assistant_start + len("Assistant:"):].strip()
        else:
            response = response.strip()

        end_time = time.time()
        resource_end = get_resource_usage()

        elapsed_time = end_time - start_time
        avg_cpu = (resource_start['cpu_percent'] + resource_end['cpu_percent']) / 2
        avg_memory_rss = (resource_start['memory_rss'] + resource_end['memory_rss']) / 2
        avg_memory_vms = (resource_start['memory_vms'] + resource_end['memory_vms']) / 2

        return response, elapsed_time, avg_cpu, avg_memory_rss, avg_memory_vms
    except Exception as e:
        print(f"Error generating chain of thought for prompt '{prompt}': {e}")
        elapsed_time = time.time() - start_time
        return "生成失败", elapsed_time, 0, 0, 0

# 遍历 DataFrame 并生成思维链
rows_list = []  # 存储每个循环产生的数据
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    description = row['描述']
    if pd.isna(description):  # 跳过空值
        continue

    # 生成思维链及资源信息
    chain_of_thought, elapsed_time, cpu_usage, memory_rss, memory_vms = generate_chain_of_thought(description)

    # 更新原始 DataFrame 中的数据
    df.at[idx, '思维链'] = chain_of_thought

    # 将资源使用情况添加到列表中
    rows_list.append({
        '描述': description,
        '处理时间': elapsed_time,
        'CPU使用率': cpu_usage,
        '内存使用RSS': memory_rss,
        '内存使用VMS': memory_vms
    })

    # 验证正则表达式是否被修改
    original_regex = row['正则表达式']
    if pd.notna(original_regex) and original_regex in chain_of_thought:
        print(f"第 {idx + 1} 行的正则表达式未被修改。")
    else:
        print(f"警告：第 {idx + 1} 行的正则表达式可能被修改，请检查！")

# 将所有资源使用情况一次性合并到 DataFrame 中
resource_df = pd.concat([resource_df, pd.DataFrame(rows_list)], ignore_index=True)

# 保存结果到新的 Excel 文件
df.to_excel(output_file, index=False)
print(f"处理完成，原始数据已保存到 {output_file}")

# 保存资源使用情况和时间信息到单独的 Excel 文件
resource_df.to_excel(resource_usage_file, index=False)
print(f"资源使用情况和时间信息已保存到 {resource_usage_file}")