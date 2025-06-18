import pandas as pd
import re
from collections import defaultdict
import random
import logging
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("self_consistency_processing.log"),
        logging.StreamHandler()
    ]
)

# =============== 正则表达式标准化 & 等价性判断函数 ===============
def normalize_regex(regex_str):
    if not isinstance(regex_str, str) or not regex_str.strip():
        return regex_str

    # 标准化字符类
    regex_str = regex_str.replace('[0-9]', '\\d')
    regex_str = regex_str.replace('[a-zA-Z]', '[A-Za-z]')
    regex_str = regex_str.replace('[^0-9]', '\\D')
    regex_str = regex_str.replace('[^a-zA-Z]', '[^A-Za-z]')
    regex_str = regex_str.replace('[ \t\n\r\f\v]', '\\s')
    regex_str = regex_str.replace('[^ \t\n\r\f\v]', '\\S')

    # 处理量词
    regex_str = re.sub(r'\{1\}', '', regex_str)
    regex_str = re.sub(r'\{1,1\}', '', regex_str)
    regex_str = regex_str.replace('{1,}', '+')

    # 去除多余转义
    regex_str = regex_str.replace('\\\\d', '\\d').replace('\\\\w', '\\w')

    # 统一括号写法
    regex_str = regex_str.replace('(?:', '(').replace(')', ')')

    # 移除边界符号
    if regex_str.startswith('^'):
        regex_str = regex_str[1:]
    if regex_str.endswith('$'):
        regex_str = regex_str[:-1]

    # 去除空格
    regex_str = regex_str.replace(' ', '')

    return regex_str


def are_regex_equivalent(regex1, regex2):
    try:
        r1 = Regex(normalize_regex(regex1))
        r2 = Regex(normalize_regex(regex2))

        dfa1 = r1.to_epsilon_nfa().to_deterministic().minimize()
        dfa2 = r2.to_epsilon_nfa().to_deterministic().minimize()

        return dfa1.is_equivalent_to(dfa2)

    except Exception as e:
        logging.warning(f"无法将正则表达式转换为 DFA 进行比较: {e}")
        test_strings = [
            '', '0', '1', 'a', '!', '12', 'ab', 'abc123', 'special@char!', ' ' * 10,
            '1' * 10, 'hello!world', '\t', '\n', '😊'
        ]
        matches1 = [bool(re.fullmatch(regex1, s)) for s in test_strings]
        matches2 = [bool(re.fullmatch(regex2, s)) for s in test_strings]
        return matches1 == matches2


# =============== 提取最终正则表达式 ===============
def extract_final_regex(chain_of_thought):
    if pd.isna(chain_of_thought):
        return None

    lines = chain_of_thought.strip().split('\n')
    last_line = lines[-1].strip()

    match = re.search(r'正则表达式：(.*)', last_line)
    if match:
        return match.group(1).strip()
    return None


# =============== 将等价的正则表达式归类 ===============
def cluster_equivalent_regex(regex_list):
    clusters = []

    for regex in regex_list:
        matched = False
        for cluster in clusters:
            if are_regex_equivalent(regex, cluster[0]):
                cluster.append(regex)
                matched = True
                break
        if not matched:
            clusters.append([regex])
    return clusters


# =============== Self-Consistency by Equivalence Class ===============
def self_consistency_by_equivalence_class(regex_list):
    if not regex_list:
        return None

    clusters = cluster_equivalent_regex(regex_list)

    largest_cluster = max(clusters, key=len)
    logging.info(f"找到的最大类别大小为: {len(largest_cluster)}")
    logging.info(f"该类别中的正则表达式: {largest_cluster}")

    return random.choice(largest_cluster)


# =============== 主处理函数 ===============
def process_rows_with_self_consistency(df):
    df['最终正则表达式'] = None

    total_rows = len(df)
    success_count = 0

    for index, row in df.iterrows():
        regex_list = []

        for col_idx in range(2, 7):  # 第 3 到第 7 列
            chain_of_thought = row.iloc[col_idx]
            regex = extract_final_regex(chain_of_thought)
            if regex:
                regex_list.append(regex)

        if not regex_list:
            logging.warning(f"第 {index + 1} 行没有有效的正则表达式，跳过。")
            continue

        final_regex = self_consistency_by_equivalence_class(regex_list)
        df.at[index, '最终正则表达式'] = final_regex
        success_count += 1

        logging.info(f"第 {index + 1} 行处理完成，最终正则表达式: {final_regex}")

    logging.info("\n================== Self-Consistency 完成 ==================")
    logging.info(f"总处理行数: {success_count}/{total_rows}")

    return df


# =============== 主程序入口 ===============
if __name__ == "__main__":
    file_path = 'output_with_chain_of_thought.xlsx'
    df = pd.read_excel(file_path)

    df_processed = process_rows_with_self_consistency(df)

    new_file_path = 'output_with_final_regex_self_consistency.xlsx'
    df_processed.to_excel(new_file_path, index=False)

    logging.info(f"处理完成，已将最终正则表达式保存到新文件：{new_file_path}")