import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import psutil

# 强制使用 GPU 0
torch.cuda.set_device(0)

# 定义量化配置（可选：如果你也需要显存优化）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 模型路径
model_dir = './output/final_model'
base_model_name = "Qwen/Qwen2.5-14B-Instruct"

# 加载基础模型并应用量化
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# 加载 LoRA 微调权重
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")

# 文件路径
input_file = './data/test_data.xlsx'
output_file = './data/output_with_chain_of_thought.xlsx'
resource_usage_file = './data/resource_usage.xlsx'  # 资源记录文件路径

# 读取数据
df = pd.read_excel(input_file)

# 检查必要列
if '描述' not in df.columns:
    raise ValueError("Excel 文件中缺少 '描述' 列")
if '正则表达式' not in df.columns:
    raise ValueError("Excel 文件中缺少 '正则表达式' 列")
for col in ['思维链1', '思维链2', '思维链3', '思维链4', '思维链5']:
    if col not in df.columns:
        df[col] = None

# ========== 添加：获取当前进程资源使用的函数 ==========
def get_resource_usage():
    process = psutil.Process()
    cpu_percent = process.cpu_percent(interval=0.1) / psutil.cpu_count()  # 单位 %
    memory_info = process.memory_info()
    return {
        'cpu_percent': cpu_percent,
        'memory_rss': memory_info.rss / (1024 * 1024),  # 转换为 MB
        '内存_vms': memory_info.vms / (1024 * 1024)   # 转换为 MB
    }

# ========== 添加：创建资源记录 DataFrame ==========
resource_df = pd.DataFrame(columns=['描述', '处理时间', 'CPU使用率', '内存使用RSS', '内存使用VMS'])

# ========== 修改：生成多个思维链 + 时间与资源统计 ==========
def generate_multiple_chains(prompt):
    start_time = time.time()
    resource_start = get_resource_usage()

    try:
        full_text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(full_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to("cuda")

        generated_responses = []
        for _ in range(5):  # 生成 5 条思维链
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=5,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_start = response.find("Assistant:")
            if assistant_start != -1:
                response = response[assistant_start + len("Assistant:"):].strip()
            else:
                response = response.strip()
            generated_responses.append(response)

        end_time = time.time()
        resource_end = get_resource_usage()

        elapsed_time = end_time - start_time
        avg_cpu = (resource_start['cpu_percent'] + resource_end['cpu_percent']) / 2
        avg_memory_rss = (resource_start['memory_rss'] + resource_end['memory_rss']) / 2
        avg_memory_vms = (resource_start['内存_vms'] + resource_end['内存_vms']) / 2

        return generated_responses, elapsed_time, avg_cpu, avg_memory_rss, avg_memory_vms
    except Exception as e:
        print(f"Error generating chains for prompt '{prompt}': {e}")
        elapsed_time = time.time() - start_time
        return ["生成失败"] * 5, elapsed_time, 0, 0, 0

# ========== 主循环中新增资源信息收集 ==========
rows_list = []  # 存储每个样本的资源信息
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    description = row['描述']
    if pd.isna(description):
        continue

    chains, elapsed_time, cpu_usage, memory_rss, memory_vms = generate_multiple_chains(description)

    # 更新思维链列
    for i, chain in enumerate(chains):
        df.at[idx, f'思维链{i+1}'] = chain

    # 验证正则表达式是否被修改
    original_regex = row['正则表达式']
    if pd.notna(original_regex):
        for i, chain in enumerate(chains):
            if original_regex in chain:
                print(f"第 {idx + 1} 行的正则表达式在思维链{i+1}中未被修改。")
            else:
                print(f"警告：第 {idx + 1} 行的正则表达式可能在思维链{i+1}中被修改，请检查！")

    # 收集资源信息
    rows_list.append({
        '描述': description,
        '处理时间': elapsed_time,
        'CPU使用率': cpu_usage,
        '内存使用RSS': memory_rss,
        '内存使用VMS': memory_vms
    })

# ========== 保存资源信息到 Excel ==========
resource_df = pd.concat([resource_df, pd.DataFrame(rows_list)], ignore_index=True)
resource_df.to_excel(resource_usage_file, index=False)
print(f"资源使用情况和时间信息已保存到 {resource_usage_file}")

# 保存原始输出文件
df.to_excel(output_file, index=False)
print(f"处理完成，结果已保存到 {output_file}")