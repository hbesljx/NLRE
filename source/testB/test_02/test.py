from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# 单次对话用于测试微调后的大模型是否可以正常使用

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

# 定义一个函数来生成输出
def generate_output(prompt):
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

        # 打印输入张量（调试用）
        print(f"Inputs: {inputs}")

        # 使用模型生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 最大生成 512 个新 token
                num_beams=5,         # 使用 beam search
                do_sample=False      # 使用贪婪解码
            )

        # 打印输出张量（调试用）
        print(f"Outputs: {outputs}")

        # 解码生成的输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 "Assistant:" 后的内容
        assistant_start = response.find("Assistant:")
        if assistant_start != -1:
            response = response[assistant_start + len("Assistant:"):].strip()
        else:
            response = response.strip()

        return response
    except Exception as e:
        print(f"Error generating output for prompt '{prompt}': {e}")
        return "生成失败"

# 测试输入数据
# test_input = "匹配以hjmozom结尾且总长度为7-9的内容"
# test_input = "满足由8个单词构成的句子"
# test_input = "满足由8个单词组成的句子 可能包括"
# test_input = "匹配不超过n位数字"
test_input = "匹配由2-9个单词组成的句子，每个单词长度为5-7"

# 调用模型生成输出
output = generate_output(test_input)

# 打印生成的输出
print(f"Input: {test_input}")
print(f"Output: {output}")