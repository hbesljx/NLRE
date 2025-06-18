import pandas as pd
import re

# 读取 Excel 文件
file_path = 'output_with_chain_of_thought.xlsx'
df = pd.read_excel(file_path)


# 定义一个函数来检查并删除包含多个“组合以上步骤”的行
def filter_rows_with_multiple_combinations(df):
    """
    删除第三列中包含多个“组合以上步骤”的行。
    """
    # 初始化需要保留的索引列表
    indices_to_keep = []

    for index, row in df.iterrows():
        col3 = row.iloc[2]  # 第三列
        if pd.isna(col3):  # 如果第三列为空，跳过
            continue

        # 计算“组合以上步骤”出现的次数
        combination_count = col3.count("组合以上步骤")

        # 如果出现次数小于等于1，则保留该行
        if combination_count <= 1:
            indices_to_keep.append(index)

    # 根据保留的索引筛选数据
    df_filtered = df.loc[indices_to_keep].reset_index(drop=True)
    return df_filtered


# 定义一个函数来提取最后一行的最终正则表达式
def extract_final_regex(chain_of_thought):
    """
    提取思维链中的最终正则表达式。
    """
    if pd.isna(chain_of_thought):  # 如果内容为空，返回 None
        return None

    # 按行分割内容
    lines = chain_of_thought.strip().split('\n')

    # 提取最后一行
    last_line = lines[-1].strip()

    # 匹配最后一行中“正则表达式：”后面的所有内容
    match = re.search(r'正则表达式：(.*)', last_line)
    if match:
        return match.group(1).strip()  # 提取并去除多余空格
    return None  # 如果没有匹配到，返回 None


# 数据预处理：移除包含多个“组合以上步骤”的行
df = filter_rows_with_multiple_combinations(df)

# 应用函数提取第三列中的最终正则表达式，并保存到第四列
df['最终正则表达式'] = df.iloc[:, 2].apply(extract_final_regex)

# 保存到新文件
new_file_path = 'output_with_final_regex.xlsx'
df.to_excel(new_file_path, index=False)

print(f"处理完成，已将最终正则表达式保存到新文件：{new_file_path}")