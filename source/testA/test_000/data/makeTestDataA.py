import random
import datetime
import re
from openpyxl import Workbook

# 假设已经导入了jieba分词库
import jieba

# ==================== 数据定义 ====================
# 扩充后的 custom_synonyms
custom_synonyms = {
    "数字": ["数值", "数位", "数码"],
    "匹配": ["验证", "检测", "识别", "比对"],
    "文件": ["文档", "档案", "资料", "记录"],
    "日期": ["时间", "年月日", "时段", "时刻"],
    "正则表达式": ["规则表达式", "模式表达式", "匹配规则", "正则"],
    "包含": ["含有", "涵盖", "带有", "具备"],
    "开头": ["起始", "开始", "前缀"],
    "结尾": ["末尾", "结束", "后缀"],
    "颜色": ["色彩", "色调", "色码"],
    "格式": ["样式", "结构", "形态"],
    "验证": ["校验", "检验", "核对"],
}

fixed_patterns = [
    {"description": "中国大陆身份证号(18位)", "regex": r"^\d{17}[\dXx]$"},
    {"description": "电子邮箱(RFC 5322)", "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
    {"description": "IPv4地址", "regex": r"^(25[0-5]|2[0-4]\d|[01]?\d\d?)\.(25[0-5]|2[0-4]\d|[01]?\d\d?)\.(25[0-5]|2[0-4]\d|[01]?\d\d?)\.(25[0-5]|2[0-4]\d|[01]?\d\d?)$"},
]

original_patterns = [
    {
        "name": "digit_range",
        "description": "匹配{min}-{max}位数字",
        "regex": r"^\d{{{min},{max}}}$",
        "placeholders": {
            "min": {"type": "range", "min": 1, "max": 8},
            "max": {"type": "range", "min": 2, "max": 16}  # 初始先设一个合法范围
        }
    },
    {
        "name": "prefix_string",
        "description": "匹配以{prefix}开头的字符串",
        "regex": r"^{prefix}.*$",
        "placeholders": {
            "prefix": {"type": "choice", "values": ["ID_", "USER_", "DOC_", "LOG_", "DATA_", "INFO_"]}  # 新增更多前缀
        }
    },
    {
        "name": "contains_keyword",
        "description": "匹配包含{keyword}的文本",
        "regex": r"^.*{keyword}.*$",
        "placeholders": {
            "keyword": {"type": "choice", "values": ["data", "test", "example", "log", "record", "info", "file"]}  # 更多关键词
        }
    }
]

new_patterns = [
    {
        "name": "semantic_version",
        "description": "匹配{type}的{version}版本号",
        "regex": r"^{version_regex}$",
        "placeholders": {
            "type": {"type": "choice", "values": ["语义化", "数字", "传统"]},
            "version": {"type": "text", "values": ["主版本", "次版本", "补丁版本"]},
            "version_regex": {"type": "generated", "func": lambda p: (
                r"\d+\.\d+\.\d+" if p["type"] == "语义化" else 
                r"\d+\.\d+" if p["type"] == "数字" else 
                r"\d+(\.\d+)?(\.\d+)?"
            )}
        }
    },
    {
        "name": "media_file",
        "description": "匹配{format}的{media_type}文件",
        "regex": r"^{file_regex}$",
        "placeholders": {
            "media_type": {"type": "choice", "values": ["图像", "视频", "音频", "归档文件"]},
            "format": {"type": "generated", "func": lambda p: {
                "图像": ["JPEG", "PNG", "GIF", "BMP"], 
                "视频": ["MP4", "AVI", "MKV", "MOV"], 
                "音频": ["MP3", "WAV", "FLAC", "AAC"],
                "归档文件": ["ZIP", "RAR", "TAR", "7Z"]
            }[p["media_type"]] or ["TXT"]},
            "file_regex": {"type": "generated", "func": lambda p: r".*\.(" + "|".join(p["format"]).lower() + ")"}
        }
    },
    {
        "name": "color_code",
        "description": "匹配{type}的{color_mode}颜色代码",
        "regex": r"^{color_regex}$",
        "placeholders": {
            "type": {"type": "choice", "values": ["HEX", "RGB", "HSL"]},
            "color_mode": {"type": "text", "values": ["标准", "透明", "灰度"]},
            "color_regex": {"type": "generated", "func": lambda p: {
                "HEX": r"#[0-9A-Fa-f]{6}",
                "RGB": r"rgb$$1,\s*$2,\s*$3$",  # 可配合 range 类型填入具体值
                "HSL": r"hsl$$1,\s*$2%,\s*$3%$"
            }.get(p["type"], r"#[0-9A-Fa-f]{6}")}
        }
    },
    {
        "name": "date_pattern",
        "description": "匹配最近{days}天内的日期(YYYYMMDD)",
        "regex": r"({date_regex})",
        "placeholders": {
            "days": {"type": "range", "min": 7, "max": 60},  # 扩展天数
            "date_regex": {"type": "generated", "func": lambda p: generate_date_regex(p['days'])}
        }
    },
    {
        "name": "ip_address",
        "description": "匹配IPv{version}地址",
        "regex": r"^{ip_regex}$",
        "placeholders": {
            "version": {"type": "choice", "values": ["4", "6"]},
            "ip_regex": {"type": "generated", "func": lambda p: {
                "4": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
                "6": r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
            }[p["version"]]}
        }
    }
]

# ==================== 工具函数 ====================
def generate_date_regex(days):
    """生成最近N天内的日期正则表达式(YYYYMMDD格式)"""
    today = datetime.datetime.now().date()
    dates = [today - datetime.timedelta(days=i) for i in range(days)]
    date_strs = [d.strftime("%Y%m%d") for d in dates]
    
    regex_parts = []
    for y in set(d[:4] for d in date_strs):
        months = set(m[4:6] for m in date_strs if m.startswith(y))
        month_day_parts = []
        for m in months:
            days_in_month = [d[6:] for d in date_strs if d.startswith(f"{y}{m}")]
            if days_in_month:
                month_day_parts.append(f"{m}({'|'.join(days_in_month)})")
        regex_parts.append(f"{y}({'|'.join(month_day_parts)})")
    return "|".join(regex_parts)

def replace_placeholders(pattern):
    placeholder_values = {}
    pattern_name = pattern.get("name", "Unnamed Pattern")

    if "placeholders" in pattern:
        for name, ph_config in pattern["placeholders"].items():
            try:
                if ph_config["type"] == "choice":
                    val = random.choice(ph_config["values"])
                elif ph_config["type"] == "range":
                    # 特殊处理 max: 确保大于当前已知的 min 值
                    if name == "max" and "min" in placeholder_values:
                        ph_config["min"] = max(ph_config["min"], placeholder_values["min"] + 1)
                    val = random.randint(ph_config["min"], ph_config["max"])
                elif ph_config["type"] == "text":
                    val = random.choice(ph_config["values"])
                elif ph_config["type"] == "generated":
                    val = ph_config["func"](placeholder_values)
                else:
                    print(f"[警告] [{pattern_name}] 未知的占位符类型: {ph_config['type']}")
                    continue
                placeholder_values[name] = val
            except Exception as e:
                print(f"[警告] [{pattern_name}] 占位符 '{name}' 生成失败: {e}")
                return None

    try:
        desc = pattern["description"].format(**placeholder_values)
        regex = pattern["regex"].format(**placeholder_values)
        return desc, regex
    except KeyError as e:
        print(f"[错误] [{pattern_name}] 替换失败 - 缺失占位符: {e}")
    except IndexError as e:
        print(f"[错误] [{pattern_name}] 正则表达式索引越界: {e}")
    return None

def synonym_replacement(text):
    words = jieba.lcut(text)
    result = []
    for word in words:
        if word in custom_synonyms and random.random() < 0.3:
            result.append(random.choice(custom_synonyms[word]))
        else:
            result.append(word)
    return "".join(result)

# ==================== 主逻辑 ====================
def generate_dataset(size=300, max_attempts=20000):
    """生成数据集（带最大尝试次数保护）"""
    results = set()
    
    # 添加固定模式
    for fp in fixed_patterns:
        results.add((fp["description"], fp["regex"]))
    
    all_patterns = original_patterns + new_patterns
    weight_original = [0.3] * len(original_patterns)
    weight_new = [0.7] * len(new_patterns)
    weights = weight_original + weight_new

    attempt = 0
    while len(results) < size and attempt < max_attempts:
        attempt += 1
        pattern = random.choices(all_patterns, weights=weights, k=1)[0]
        res = replace_placeholders(pattern)
        if res:
            desc, regex = res
            desc = synonym_replacement(desc)
            results.add((desc, regex))

    if attempt >= max_attempts:
        print(f"[警告] 达到最大尝试次数 {max_attempts}，未能生成足够数量的数据")
    
    return [{"描述": d, "正则表达式": r} for d, r in results]

def save_to_excel(data, filename="test_dataA.xlsx"):
    wb = Workbook()
    ws = wb.active
    ws.append(["描述", "正则表达式"])
    for item in data:
        ws.append([item["描述"], item["正则表达式"]])
    wb.save(filename)

if __name__ == "__main__":
    dataset = generate_dataset(300)
    save_to_excel(dataset)
    print(f"已生成 {len(dataset)} 条正则表达式")
    for i in range(3):  # 打印3条示例
        print(f"{dataset[i]['描述']} => {dataset[i]['正则表达式']}")