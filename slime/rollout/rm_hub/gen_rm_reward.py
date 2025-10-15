from collections import Counter
from typing import Literal
from itertools import combinations
import math
import re, json
from typing import List, Dict, Union


def extract_score_list(response_text: str) -> List[Dict[str, Union[int, float]]]:
    """
    从整段文本中提取 [{"helpfulness":..,"format":..}, ...]，若不存在则返回 []。
    分数字段统一转为 int/float（字符串如 "8.5分" 也尽量解析）。
    """

    def _to_number(x):
        # 把任意值转成 int/float；失败返回 None
        if isinstance(x, (int, float)):
            return int(x) if isinstance(x, float) and x.is_integer() else x
        if isinstance(x, str):
            s = x.strip().strip('"').strip("'")
            if re.fullmatch(r"[+-]?\d+", s):
                return int(s)
            try:
                v = float(s)
                return int(v) if v.is_integer() else v
            except ValueError:
                m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
                if m:
                    v = float(m.group(0))
                    return int(v) if v.is_integer() else v
        return None

    t = response_text or ""
    # 统一引号、去掉特殊标记
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r"<\|?endoftext\|?>", "", t, flags=re.I)
    # 找出所有数组片段，筛选含两个字段的
    candidates = []
    for m in re.finditer(r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]", t, flags=re.S):
        block = m.group(0)
        if re.search(r'"helpfulness"\s*:', block) and re.search(r'"format"\s*:', block):
            candidates.append(block)
    if not candidates:
        return []
    # 取最长候选，一般为目标；清理尾逗号
    json_text = re.sub(r",(\s*[}\]])", r"\1", max(candidates, key=len))
    # 先尝试标准 JSON 解析
    try:
        arr = json.loads(json_text)
        if not isinstance(arr, list):
            return []
        out = []
        for item in arr:
            if isinstance(item, dict):
                hv = _to_number(item.get("helpfulness"))
                fv = _to_number(item.get("format"))
                if hv is not None and fv is not None:
                    out.append({"helpfulness": hv, "format": fv})
        return out
    except Exception:
        # 兜底：任意顺序提取对象内的两个字段
        obj_pat = re.compile(
            r"\{[^{}]*?(?:"
            r'"helpfulness"\s*:\s*([^,}]+)[^{}]*?"format"\s*:\s*([^,}]+)'
            r"|"
            r'"format"\s*:\s*([^,}]+)[^{}]*?"helpfulness"\s*:\s*([^,}]+)'
            r")[^{}]*?\}",
            re.S,
        )
        out = []
        for g1, g2, g3, g4 in obj_pat.findall(json_text):
            hv_raw, fv_raw = (g1, g2) if g1 else (g4, g3)
            hv, fv = _to_number(hv_raw), _to_number(fv_raw)
            if hv is not None and fv is not None:
                out.append({"helpfulness": hv, "format": fv})
        return out


def _tie_pairs(lst):
    """返回列表中'并列对'的数量：对每个取值频次 c，贡献 c*(c-1)/2 个并列对。"""
    cnt = Counter(lst)
    return sum(c * (c - 1) // 2 for c in cnt.values())


def _kendall_counts(y_true, y_pred):
    """
    C: 一致对, D: 不一致对
    Tx: 仅在 y_true 中并列的对数
    Ty: 仅在 y_pred  中并列的对数
    （两边都并列的对跳过，不计入这四项）
    """
    C = D = Tx = Ty = 0
    n = len(y_true)
    for i, j in combinations(range(n), 2):
        dx = (y_true[i] > y_true[j]) - (y_true[i] < y_true[j])
        dy = (y_pred[i] > y_pred[j]) - (y_pred[i] < y_pred[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0 and dy != 0:
            Tx += 1
        elif dx != 0 and dy == 0:
            Ty += 1
        elif dx == dy:
            C += 1
        else:
            D += 1
    return C, D, Tx, Ty


def _kendall_tau_b(y_true, y_pred, undefined="neutral"):
    """
    正常情况返回 Kendall's tau-b。
    但在“真值或预测全并列”时：
      - 真值全并列：预测中的并列对计为正确，非并列对计为错误 → 分数 = 预测并列对 / 全部对
      - 预测全并列且真值非全并列：分数 = 真值并列对 / 全部对
      - 两边都全并列：返回 1.0（完全匹配）
    其他分母为0的极端情况：按 undefined 策略处理（默认 0.0）。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    if n < 2:
        return 0.0  # 没有可比较对，返回中性
    total_pairs = n * (n - 1) // 2
    gt_all_tie = len(set(y_true)) == 1
    pred_all_tie = len(set(y_pred)) == 1
    if gt_all_tie:
        # 预测并列对算正确，其余算错误
        return _tie_pairs(y_pred) / total_pairs
    if pred_all_tie:
        # 预测全并列但真值有顺序：只把真值中的并列对算正确
        return _tie_pairs(y_true) / total_pairs
    # ---- 正常 tau-b ----
    C, D, Tx, Ty = _kendall_counts(y_true, y_pred)
    denom = math.sqrt((C + D + Tx) * (C + D + Ty))
    if denom == 0:
        # 极端：没有任一对能在两边同时比较出高低（很少见）
        if undefined == "neutral":
            return 0.0
        elif undefined == "nan":
            return float("nan")
        elif undefined == "error":
            raise ValueError("Kendall tau-b undefined: denominator=0.")
        else:
            return 0.0
    return (C - D) / denom


def _goodman_kruskal_gamma(y_true, y_pred, undefined="neutral"):
    """相当于忽视结影响的_kendall_tau_b"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    if n < 2:
        return 0.0  # 没有可比较对，返回中性
    total_pairs = n * (n - 1) // 2
    gt_all_tie = len(set(y_true)) == 1
    pred_all_tie = len(set(y_pred)) == 1
    if gt_all_tie:
        # 预测并列对算正确，其余算错误
        return _tie_pairs(y_pred) / total_pairs
    if pred_all_tie:
        # 预测全并列但真值有顺序：只把真值中的并列对算正确
        return _tie_pairs(y_true) / total_pairs
    # ---- 正常 tau-b ----
    C, D, _, _ = _kendall_counts(y_true, y_pred)
    denom = C + D
    if denom == 0:
        # 极端：没有任一对能在两边同时比较出高低（很少见）
        if undefined == "neutral":
            return 0.0
        elif undefined == "nan":
            return float("nan")
        elif undefined == "error":
            raise ValueError("Kendall tau-b undefined: denominator=0.")
        else:
            return 0.0
    return (C - D) / denom


def comput_abs_score(
    ground_truth_list, pred_list, margin=1, mode: Literal["mean_score", "acc_within_margin"] = "mean_score"
):
    """
    可以有两个思路：1.预测得分在margin内的为1，没有在margin的为0, 即mode: acc_within_margin
    2. 在margin内的视为命中。不在margin内的，预测差异越大得分越小,即mode： mean_score
    思路2可能存在问题是，可能倾向于预测高频率的得分，而不是真正学质量打分。
    """
    if len(ground_truth_list) != len(pred_list):
        raise ValueError("ground_truth_list 与 pred_list 长度必须一致。")
    if len(ground_truth_list) == 0:
        raise ValueError("输入列表不能为空。")
    if margin < 0:
        raise ValueError("margin 必须为非负数。")
    diffs = [abs(g - p) for g, p in zip(ground_truth_list, pred_list)]
    n = len(diffs)
    if mode == "acc_within_margin":
        return sum(d <= margin for d in diffs) / n
    # mode == "mean_score"
    adj = [min(max(0.0, d - margin), 10) for d in diffs]
    scores = [1.0 - 0.2 * a for a in adj]
    return sum(scores) / n


def comput_preference_score(ground_truth_list, pred_list, metric="tau_b"):
    """
    这个是很经典的顺序比较，有比较多可能的指标
    - "tau_b" (默认) | "kendall_tau_b"
           - "c_index"
           - "gamma"  (Goodman–Kruskal)
           - "spearman"
           - "footrule" | "footrule_norm"
           - "kendall_dist" | "kendall_dist_norm"
           - "ndcg"
    """
    if len(ground_truth_list) != len(pred_list):
        raise ValueError("ground_truth_list 与 pred_list 长度不一致")
    n = len(ground_truth_list)
    if n < 2:
        return 0.0
    y_true = list(ground_truth_list)
    y_pred = list(pred_list)
    # 解析 "ndcg@k" 形式
    m = metric.lower()
    aliases = {
        "kendall_tau_b": "tau_b",
        "tau-b": "tau_b",
        "tau_b": "tau_b",
        "c-index": "c_index",
        "c_index": "c_index",
        "gamma": "gamma",
        "goodman_kruskal_gamma": "gamma",
        "spearman": "spearman",
        "rho": "spearman",
        "footrule": "footrule",
        "footrule_norm": "footrule_norm",
        "kendall_dist": "kendall_dist",
        "kendall_dist_norm": "kendall_dist_norm",
        "ndcg": "ndcg",
    }
    key = aliases.get(m, None)
    if key is None:
        raise ValueError(f"未知 metric: {metric}")
    if key == "tau_b":
        return _kendall_tau_b(y_true, y_pred)
    elif key == "gamma":
        return _goodman_kruskal_gamma(y_true, y_pred)
    else:
        raise RuntimeError("未实现的 metric")


def gen_rm_reward(response: str, label: List[Dict[str, Union[int, float]]]) -> float:
    """
    合并 helpfulness(0.7) 与 format(0.3) 得到预测分，和 label['score'] 对比。
    所有中间值与最终值均四舍五入保留 1 位小数。
    若数量不一致返回 -10.0
    依赖: extract_score_list, comput_abs_score, comput_preference_score
    """

    def round_num(x, digits=1, adjust_factor=1.0):
        """四舍五入，默认保留1位小数。adjust_factor用来更精细的倍数（比如是2的话就是四舍五入到x.5）"""
        if isinstance(x, str):
            x = float(x)
        if not isinstance(x, (int, float)):
            raise ValueError(f"无法对非数字类型 {type(x)} 进行四舍五入")
        if math.isnan(x) or math.isinf(x):
            raise ValueError(f"无法对{x}进行四舍五入")
        factor = 10**digits
        return int(x * factor * adjust_factor + 0.5) / factor / adjust_factor

    pred_list = extract_score_list(response)
    if len(pred_list) != len(label):
        return -10.0, -10.0, -10.0
    try:
        merge_pred_score = [round_num(0.7 * p["helpfulness"] + 0.3 * p["format"]) for p in pred_list]
        merge_label_score = [round_num(d["score"]) for d in label]
        merged_pred_pref_score = [round_num(s, digits=0, adjust_factor=2) for s in merge_pred_score]
        merged_label_pref_score = [round_num(s, digits=0, adjust_factor=2) for s in merge_label_score]
        abs_score = comput_abs_score(merge_pred_score, merge_label_score, mode="mean_score", margin=0.5)
        perf_score = comput_preference_score(merged_pred_pref_score, merged_label_pref_score, metric="gamma")
        final_score = round_num(0.6 * abs_score + 0.4 * perf_score, digits=2) if len(pred_list) >= 2 else abs_score
        return final_score, abs_score, perf_score
    except Exception as e:
        print(f"Error when getting reward scores: {e}")
        return -10.0, -10.0, -10.0


if __name__ == "__main__":
    gt = [10, 8]
    pred = [10, 10]
    print(comput_abs_score(gt, pred, margin=1))  # 默认 mean_score
    print(comput_abs_score(gt, pred, margin=1, mode="acc_within_margin"))  # 命中率
    print(comput_preference_score(gt, pred, metric="tau_b"))
    print(comput_preference_score(gt, pred, metric="gamma"))
