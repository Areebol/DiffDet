from utils import *
import torch

max_diffuse_t = 50  # ImageNet 取 50
selected_t_steps = [1, 2, 5, 10, 15, 20, 30, 50, 100]
selected_t_steps = list(filter(lambda x: x <= max_diffuse_t, selected_t_steps))


def time_change_matrix_T(score: torch.Tensor):  # (t, c, h, w)
    """计算时间变化矩阵"""
    return torch.diff(score, dim=0)  # (t-1, c, h, w)


def cumulative_trend_vector_C(score: torch.Tensor):  # (t, c, h, w)
    """计算累积趋势向量"""
    return torch.sum(score, dim=0, keepdim=True) / score.shape[0]  # (1, c, h, w)


def score_direction_consistency_matrix_D(
    score: torch.Tensor,  # (t, c, h, w)
) -> torch.Tensor:
    """
    计算 Score 方向一致性矩阵 D(score)，使用 RGB 通道的余弦相似度，输出维度为 (t-1, c, h, w)。
    注意：这里将像素级的余弦相似度标量复制到所有通道。

    参数:
        score: 输入向量 (score)，形状为 (t, c, h, w)，假设 c=3 (RGB)，torch.Tensor 类型。

    返回值:
        Score 方向一致性矩阵 D(score)，形状为 (t-1, c, h, w)，torch.Tensor 类型。
    """
    t, c, h, w = score.shape

    score_t = score[:-1]  # 形状 (t-1, c, h, w)
    score_t_plus_1 = score[1:]  # 形状 (t-1, c, h, w)

    cos_sim = F.cosine_similarity(
        score_t, score_t_plus_1, dim=1, eps=1e-8
    )  # 形状 (t-1, h, w)

    # 将余弦相似度复制到所有通道
    return cos_sim.unsqueeze(1).repeat(1, c, 1, 1)  # 形状 (t-1, c, h, w)


def score_feature_at_timesteps(scores_at_t: torch.Tensor) -> dict[int, torch.Tensor]:
    """计算不同时间步的视频分数

    Args:
        scores_at_t: 形状为 (t_steps, num_frames, channels, height, width) 的张量,
            表示每个时间步的 Score

    Returns:
        dict: key 为时间步, value 为对应时间步的特征张量 (2t-1, c, h, w)
    """
    # 对前 selected_t_steps 分别求平均
    score_feature_at_timestep = {}

    for t_step in selected_t_steps:
        debug(f"t_step: {t_step}")

        # 取出当前时间步的分数
        score = scores_at_t[t_step - 1]  # (t, c, h, w)
        debug(f"score: {tensor_detail(score)}")

        # 计算时序特征
        feature_T = time_change_matrix_T(score)  # (t-1, c, h, w)
        debug(f"feature_T: {tensor_detail(feature_T)}")

        feature_C = cumulative_trend_vector_C(score)  # (1, c, h, w)
        debug(f"feature_C: {tensor_detail(feature_C)}")

        feature_D = score_direction_consistency_matrix_D(score)  # (t-1, c, h, w)
        debug(f"feature_D: {tensor_detail(feature_D)}")

        # concat time series related feature T, C, D
        feature = torch.cat([feature_T, feature_C, feature_D], dim=0)  # (2t-1, c, h, w)
        debug(f"feature: {tensor_detail(feature)}")

        score_feature_at_timestep[t_step] = feature

    return score_feature_at_timestep


# 示例使用 (假设你的输入数据为 x)
if __name__ == "__main__":
    # 假设我们有一个随机的输入向量，形状为 (t, c, h, w)
    s, t, c, h, w = 50, 8, 3, 224, 224
    scores_at_t = torch.randn(s, t, c, h, w)

    score_at_t = scores_at_t[0]
    T_matrix = time_change_matrix_T(score_at_t)
    print("Time Change Matrix T(x) shape:", T_matrix.shape)  # 预期形状: (32, 32, 4)

    C_vector = cumulative_trend_vector_C(score_at_t)
    print(
        "Cumulative Trend Vector C(x) shape:", C_vector.shape
    )  # 预期形状: (1, 32, 32)

    D_matrix = score_direction_consistency_matrix_D(score_at_t)
    print(
        "Score Direction Consistency Matrix D(x) shape:", D_matrix.shape
    )  # 预期形状: (32, 32, 4)

    score_feature_at_timestep = score_feature_at_timesteps(scores_at_t)
    print(list(score_feature_at_timestep.keys()))
    print(list(score_feature_at_timestep.values())[0].shape)
