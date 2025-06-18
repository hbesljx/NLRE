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


def normalize_regex(regex_str):
    """
    标准化正则表达式：
    1. 将 {1,} 替换为 +。
    2. 移除边界符号 ^ 和 $（如果它们分别位于正则表达式的开头和结尾）。
    3. 处理其他标准化操作。
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
    regex_str = regex_str.replace('{1,}', '+')             # {1,} -> +

    # 3. 去除多余的转义字符
    regex_str = regex_str.replace('\\\\d', '\\d')
    regex_str = regex_str.replace('\\\\w', '\\w')

    # 4. 统一括号和分组的写法
    regex_str = regex_str.replace('(?:', '(')
    regex_str = regex_str.replace(')', ')')

    # 5. 移除边界符 ^ 和 $
    if regex_str.startswith('^'):
        regex_str = regex_str[1:]  # 移除 ^
    if regex_str.endswith('$'):
        regex_str = regex_str[:-1]  # 移除 $

    # 6. 移除多余的空格
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
        # 正确转义括号
        regex_str = (
            regex_str
            .replace('(', r'$')
            .replace(')', r'$')
            .replace('.', r'\.')
            .replace('*', r'\*')
            .replace('+', r'\+')
            .replace('?', r'\?')
        )

        regex = Regex(regex_str)
        min_dfa = regex.to_epsilon_nfa().to_deterministic().minimize()
        return min_dfa
    except Exception as e:
        logging.error(f"无法将正则表达式转换为 DFA: '{regex_str}'，错误: {e}")
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
    """
    判断两个正则表达式是否等价。
    
    参数:
        regex1 (str): 第一个正则表达式。
        regex2 (str): 第二个正则表达式。
        log_details (bool): 是否打印详细的日志信息，默认为 False。
        
    返回:
        bool: 如果两个正则表达式等价，则返回 True；否则返回 False。
    """
    # 标准化正则表达式
    regex1 = normalize_regex(regex1)
    regex2 = normalize_regex(regex2)

    logging.debug(f"正在比较正则表达式: '{regex1}' vs '{regex2}'")

    # 尝试使用 DFA 方法判断等价性
    try:
        dfa1 = regex_to_min_dfa(regex1)
        dfa2 = regex_to_min_dfa(regex2)
        if dfa1 is not None and dfa2 is not None:
            is_equivalent = dfa1.is_equivalent_to(dfa2)
            if log_details:
                logging.info(f"DFA 比较结果：'{regex1}' 和 '{regex2}' {'等价' if is_equivalent else '不等价'}")
            return is_equivalent
        else:
            logging.warning("DFA 构造失败，将使用备用方法进行比较。")
    except Exception as e:
        logging.warning(f"DFA 方法失败（可能因为无效正则表达式）：{e}")

    # 备用方案 1：使用 rstr 生成随机字符串比较
    try:
        if are_regex_equivalent_by_rstr(regex1, regex2):
            if log_details:
                logging.info(f"通过 rstr 随机字符串判断：'{regex1}' 和 '{regex2}' 等价")
            return True
    except Exception as e:
        logging.warning(f"rstr 方法失败：{e}")

    # 备用方案 2：使用测试字符串集合进行比较
    test_strings = generate_test_strings()
    try:
        is_equivalent = are_regex_equivalent_by_tests(regex1, regex2, test_strings)
        if log_details:
            logging.info(f"测试字符串集比较结果：'{regex1}' 和 '{regex2}' {'等价' if is_equivalent else '不等价'}")
        return is_equivalent
    except Exception as e:
        logging.warning(f"测试字符串集方法失败：{e}")

    # 所有方法都失败时，默认视为不等价
    logging.error(f"所有方法均无法判断等价性，最终视为不等价：'{regex1}' vs '{regex2}'")
    return False


# 计算编辑距离并归一化为相似度分数
def calculate_similarity(regex1, regex2):
    edit_distance = Levenshtein.distance(regex1, regex2)
    max_length = max(len(regex1), len(regex2))
    normalized_edit_distance = edit_distance / max_length if max_length > 0 else 0
    similarity_score = 1 - normalized_edit_distance
    return similarity_score


# 主程序
# 主程序
if __name__ == "__main__":
    # 创建保存目录
    os.makedirs("./part_logs", exist_ok=True)

    # 配置日志文件
    log_file = "./part_logs/regex_comparison.log"
    setup_logging(log_file)

    # 读取Excel文件
    file_path = "./data/output_with_final_regex_self_consistency.xlsx"
    df = pd.read_excel(file_path)

    # 检查是否有足够的列
    if df.shape[1] < 8:
        raise ValueError("Excel文件中列数不足，请确保至少有8列。")

    # 初始化统计变量
    total_rows = len(df)
    equivalent_count = 0
    partial_match_count = 0
    results = []

    # 遍历每一行并比较第二列和第八列的正则表达式
    for index, row in df.iterrows():
        regex_col2 = row.iloc[1]  # 第二列
        regex_col8 = row.iloc[7]  # 第八列

        if pd.isna(regex_col2) or pd.isna(regex_col8):
            logging.warning(f"第 {index + 1} 行有空值，跳过比较。")
            continue

        # 比较两个正则表达式是否等价
        is_equivalent = are_regex_equivalent(regex_col2, regex_col8, log_details=True)
        if is_equivalent:
            equivalent_count += 1
            partial_match_count += 1  # 等价的也一定是部分匹配
        else:
            # 计算编辑距离相似度分数
            similarity_score = calculate_similarity(regex_col2, regex_col8)
            if similarity_score >= 0.8:  # 设定部分匹配的阈值为 0.8
                partial_match_count += 1

        # 记录每行的结果
        results.append({
            "Row": index + 1,
            "Regex1": regex_col2,
            "Regex2": regex_col8,
            "IsEquivalent": is_equivalent,
            "SimilarityScore": similarity_score if not is_equivalent else 1.0  # 等价时相似度视为1.0
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