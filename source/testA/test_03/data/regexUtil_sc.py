import pandas as pd
import re
from collections import Counter

# 读取 Excel 文件
file_path = 'output_with_chain_of_thought.xlsx'
df = pd.read_excel(file_path)

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


# 定义 Self-Consistency 函数
def self_consistency(regex_list):
    """
    对多个正则表达式进行 Self-Consistency 处理，选择出现次数最多的正则表达式。
    如果所有正则表达式都只出现一次，则返回第一个。
    """
    if not regex_list:  # 如果列表为空，返回 None
        return None

    # 使用 Counter 统计每个正则表达式的出现次数
    counter = Counter(regex_list)

    # 找到出现次数最多的正则表达式
    most_common_regex, count = counter.most_common(1)[0]

    # 如果出现次数大于 1，返回该正则表达式；否则返回第一个
    return most_common_regex if count > 1 else regex_list[0]


# 数据预处理：从第 3、4、5、6、7 列提取正则表达式，并使用 Self-Consistency 处理
def process_rows_with_self_consistency(df):
    """
    从第 3、4、5、6、7 列提取正则表达式，并使用 Self-Consistency 方法获取最终结果。
    """
    # 初始化最终正则表达式列
    df['最终正则表达式'] = None

    for index, row in df.iterrows():
        regex_list = []

        # 遍历第 3、4、5、6、7 列
        for col in range(2, 7):  # 第 3 列对应索引 2，第 7 列对应索引 6
            chain_of_thought = row.iloc[col]
            regex = extract_final_regex(chain_of_thought)
            if regex:  # 如果提取到正则表达式，添加到列表
                regex_list.append(regex)

        # 使用 Self-Consistency 处理正则表达式列表
        final_regex = self_consistency(regex_list)

        # 将最终结果保存到第 8 列
        df.at[index, '最终正则表达式'] = final_regex

    return df


# 应用函数处理数据
df = process_rows_with_self_consistency(df)

# 保存到新文件
new_file_path = 'output_with_final_regex_self_consistency.xlsx'
df.to_excel(new_file_path, index=False)

print(f"处理完成，已将最终正则表达式保存到新文件：{new_file_path}")