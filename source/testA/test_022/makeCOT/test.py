from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "Qwen/Qwen2.5-14B-Instruct"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用半精度浮点数以节省显存
    device_map="auto"           # 自动分配模型到多个 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 预热对话（固定上下文）
initial_messages = [
    {"role": "system", "content": "你是千问，由阿里云创造，你是一个有用的助手"},
    {"role": "user", "content": "给我一个关于大语言模型的简短介绍."},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": '''你将根据已知的自然语言和它所描述的正则表达式来将复杂问题分解为子问题，给你一个示例如下：
    我输入：匹配电子邮件地址，它对应的正则表达式是[a-zA-Z0-9.-]+@[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*
    你输出子问题步骤链：
    step1：匹配用户名部分，对应正则表达式：[a-zA-Z0-9._-]+
    step2：匹配 @ 符号，对应正则表达式：@
    step3：匹配域名部分，对应正则表达式：[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*
    step4：组合以上步骤，对应正则表达式：[a-zA-Z0-9.-]+@[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*
    你应该严格按照这样的格式来输出并且不需要使用?: 并且你只需要给我展示子问题步骤链，而不需要输出任何其它内容'''},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "按照我的格式，即step1，step2这样来展示思维链，展示最终的子问题步骤链即可"},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "并且最后一步必须是“组合以上步骤”"},
    {"role": "assistant", "content": ""}
]

def single_talk(input_text):
    """
    单次调用模型生成响应
    :param input_text: 输入文本
    :return: 响应文本
    """
    # 构建完整的对话消息
    messages = initial_messages + [{"role": "user", "content": input_text}]

    # 构建输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 将输入转换为模型可接受的格式
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)

    # 生成输出
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        temperature=0.5,     # 调整温度
        top_k=50,            # 设置 top-k 抽样
        top_p=0.5,           # 设置 nucleus 抽样
        use_cache=True       # 禁用缓存以避免显存泄漏
    )

    # 解码生成的文本
    response = tokenizer.decode(
        generated_ids[0][len(model_inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    return response.strip()

def main():
    # 手动输入内容进行测试
    user_input = '''匹配3位数字'''
    response = single_talk(user_input)
    print(f"模型回答：\n{response}")

if __name__ == "__main__":
    main()