"""Select the task reward without owning a training loop."""
from alignment.rewards.gsm8k import numeric_reward


def reward_function(config):
    if config.reward_type == "gsm8k_numeric":
        return numeric_reward
    try:
        from alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"作业评分器缺少依赖 {error.name}；可显式使用 --reward-type gsm8k_numeric，"
            "或在获得安装授权后配置原评分器依赖"
        ) from error
    return r1_zero_reward_fn if config.reward_type == "r1_zero" else question_only_reward_fn
