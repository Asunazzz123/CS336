"""Pure numeric answer scoring; no model or training configuration."""
import re
from decimal import Decimal, InvalidOperation


def numeric_reward(response, ground_truth):
    """轻量数值评分，显式区别于原作业的数学等价评分器。"""
    # 只提取明确的最终答案位置，不用推理中最后一个数字猜测答案
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    answers = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    final = boxed[-1] if boxed else (answers[-1].strip() if answers else None)
    # r1 格式中的答案必须有正确的结束标签；boxed 位于未闭合推理中不算答案
    if "</think>" in response or "<answer>" in response:
        if "</think> <answer>" not in response or "</answer>" not in response:
            final = None
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    correct = False
    if final is not None:
        normalized = final.strip().replace(",", "")
        target = str(ground_truth).strip().replace(",", "")
        if re.fullmatch(number_pattern, normalized) and re.fullmatch(number_pattern, target):
            try:
                correct = Decimal(normalized) == Decimal(target)
            except InvalidOperation:
                pass
    return {"format_reward": float(final is not None), "answer_reward": float(correct), "reward": float(correct)}
