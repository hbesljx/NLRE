import pandas as pd
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton
import re
import rstr
import logging
import Levenshtein  # 需要安装 python-Levenshtein 库
import os


# 配置日志记录
def setup_logging(log_file):
    """
    设置日志记录器。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 将日志写入文件
            logging.StreamHandler()         # 同时输出到控制台
        ]
    )


# 正则表达式标准化
def normalize_regex(regex_str):
    """
    标准化正则表达式。
    """
    if not isinstance(regex_str, str) or not regex_str.strip():
        return regex_str  # 如果是空值或非字符串，直接返回原值

    # 1. 处理字符类
    regex_str = regex_str.replace('[0-9]', '\\d')          # [0-9] -> \d
    regex_str = regex_str.replace('[a-zA-Z]', '[A-Za-z]')  # [a-zA-Z] -> [A-Za-z]
    regex_str = regex_str.replace('[^0-9]', '\\D')         # [^0-9] -> \D
    regex_str = regex_str.replace('[^a-zA-Z]', '[^A-Za-z]')# [^a-zA-Z] -> [^A-Za-z]
    regex_str = regex_str.replace('[ \t\n\r\f\v]', '\\s')  # [ \t\n\r\f\v] -> \s
    regex_str = regex_str.replace('[^ \t\n\r\f\v]', '\\S') # [^ \t\n\r\f\v] -> \S

    # 2. 处理量词
    regex_str = re.sub(r'\{1\}', '', regex_str)            # {1} -> 空
    regex_str = re.sub(r'\{1,1\}', '', regex_str)          # {1,1} -> 空

    # 3. 去除多余的转义字符
    regex_str = regex_str.replace('\\\\d', '\\d')
    regex_str = regex_str.replace('\\\\w', '\\w')

    # 4. 统一括号和分组的写法
    # 将 (?:...) 替换为 (...)，除非明确需要非捕获分组
    regex_str = regex_str.replace('(?:', '(')
    regex_str = regex_str.replace(')', ')')

    # 5. 其他优化
    # 移除多余的空格
    regex_str = regex_str.replace(' ', '')

    return regex_str


# 生成一个用于测试的字符串列表
def generate_test_strings():
    """生成一个用于测试的字符串列表"""
    return [
        '',          # 空字符串
        '0',         # 单个数字
        '1',         # 单个数字
        'a',         # 单个字母
        '!',         # 特殊字符
        '12',        # 两个数字
        'ab',        # 两个字母
        'abc123',    # 字母和数字组合
        'special@char!',  # 包含特殊字符的字符串
        ' ' * 10,    # 多个空格
        '1' * 10,    # 多个相同数字
        'hello!world',  # 常见字符串
        '\t',        # 制表符
        '\n',        # 换行符
        '😊',        # 表情符号
    ]


# 定义一个函数将正则表达式转换为最小化的DFA
def regex_to_min_dfa(regex_str):
    try:
        # 对部分特殊字符进行转义，避免破坏正则表达式的语义
        regex_str = (
            regex_str
            .replace('(', '$')  # 转义左括号
            .replace(')', '$')  # 转义右括号
            .replace('.', '\.')  # 转义点号
            .replace('*', '\*')  # 转义星号
            .replace('+', '\+')  # 转义加号
            .replace('?', '\?')  # 转义问号
        )

        # 创建正则表达式对象
        regex = Regex(regex_str)
        # 转换为最小化的DFA
        min_dfa = regex.to_epsilon_nfa().to_deterministic().minimize()
        return min_dfa
    except Exception as e:
        logging.error(f"无法解析正则表达式 '{regex_str}'，错误: {e}")
        return None


# 使用 rstr 库生成随机字符串并比较两个正则表达式的等价性
def are_regex_equivalent_by_rstr(regex1, regex2, num_samples=100):
    try:
        # 使用 rstr 生成随机字符串
        samples1 = set(rstr.xeger(regex1) for _ in range(num_samples))
        samples2 = set(rstr.xeger(regex2) for _ in range(num_samples))

        logging.info(f"Regex1 ({regex1}) 生成的字符串: {samples1}")
        logging.info(f"Regex2 ({regex2}) 生成的字符串: {samples2}")

        # 比较两个集合是否相等
        return samples1 == samples2
    except Exception as e:
        logging.error(f"无法生成随机字符串，错误: {e}")
        return False


# 使用测试字符串集比较两个正则表达式的等价性
def are_regex_equivalent_by_tests(regex1, regex2, test_strings):
    matches1 = [bool(re.match(regex1, s)) for s in test_strings]
    matches2 = [bool(re.match(regex2, s)) for s in test_strings]

    return matches1 == matches2


# 定义一个函数比较两个正则表达式是否等价
def are_regex_equivalent(regex1, regex2, log_details=False):
    # 标准化正则表达式
    regex1 = normalize_regex(regex1)
    regex2 = normalize_regex(regex2)

    dfa1 = regex_to_min_dfa(regex1)
    dfa2 = regex_to_min_dfa(regex2)

    if dfa1 is not None and dfa2 is not None:
        # 如果两个正则表达式都能成功解析为 DFA，则直接比较
        is_equivalent = dfa1.is_equivalent_to(dfa2)
        if log_details:
            logging.info(f"DFA 比较结果：'{regex1}' 和 '{regex2}' {'等价' if is_equivalent else '不等价'}")
        return is_equivalent

    # 如果无法解析为 DFA，则尝试使用 rstr 生成随机字符串进行比较
    if are_regex_equivalent_by_rstr(regex1, regex2):
        return True

    # 最后回退到测试字符串集
    test_strings = generate_test_strings()
    is_equivalent = are_regex_equivalent_by_tests(regex1, regex2, test_strings)
    if log_details:
        logging.info(f"测试字符串集比较结果：'{regex1}' 和 '{regex2}' {'等价' if is_equivalent else '不等价'}")
    return is_equivalent


# 计算编辑距离并归一化为相似度分数
def calculate_similarity(regex1, regex2):
    edit_distance = Levenshtein.distance(regex1, regex2)
    max_length = max(len(regex1), len(regex2))
    normalized_edit_distance = edit_distance / max_length if max_length > 0 else 0
    similarity_score = 1 - normalized_edit_distance
    return similarity_score


# 主程序
if __name__ == "__main__":
    # 创建保存目录
    os.makedirs("./part_logs", exist_ok=True)

    # 配置日志文件
    log_file = "./part_logs/regex_comparison.log"
    setup_logging(log_file)

    # 读取Excel文件
    file_path = "./data/output_with_final_regex.xlsx"
    df = pd.read_excel(file_path)

    # 检查是否有足够的列
    if df.shape[1] < 4:
        raise ValueError("Excel文件中列数不足，请确保至少有4列。")

    # 初始化统计变量
    total_rows = len(df)
    equivalent_count = 0
    partial_match_count = 0
    results = []

    # 遍历每一行并比较第二列和第四列的正则表达式
    for index, row in df.iterrows():
        regex_col2 = row.iloc[1]  # 第二列
        regex_col4 = row.iloc[3]  # 第四列

        if pd.isna(regex_col2) or pd.isna(regex_col4):
            logging.warning(f"第 {index + 1} 行有空值，跳过比较。")
            continue

        # 比较两个正则表达式是否等价
        is_equivalent = are_regex_equivalent(regex_col2, regex_col4, log_details=True)
        if is_equivalent:
            equivalent_count += 1

        # 计算编辑距离相似度分数
        similarity_score = calculate_similarity(regex_col2, regex_col4)
        if similarity_score >= 0.8:  # 设定部分匹配的阈值为 0.8
            partial_match_count += 1

        # 记录每行的结果
        results.append({
            "Row": index + 1,
            "Regex1": regex_col2,
            "Regex2": regex_col4,
            "IsEquivalent": is_equivalent,
            "SimilarityScore": similarity_score
        })

    # 计算等价比例和部分匹配率
    equivalence_ratio = equivalent_count / total_rows
    partial_match_ratio = partial_match_count / total_rows
    logging.info(f"\n总行数: {total_rows}")
    logging.info(f"等价行数: {equivalent_count}")
    logging.info(f"等价比例: {equivalence_ratio:.2%}")
    logging.info(f"部分匹配行数: {partial_match_count}")
    logging.info(f"部分匹配率: {partial_match_ratio:.2%}")

    # 将结果保存到 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv("./part_logs/regex_comparison_results.csv", index=False)
    logging.info("结果已保存到 './part_logs/regex_comparison_results.csv'")