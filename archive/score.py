import random
import numpy as np
import torch
from lib.sde import VPSDE
from torch.utils.data import Dataset, DataLoader
from utils import *


diffuse_t = 50  # ImageNet 取 50
selected_t_steps = [1, 2, 5, 10, 15, 20, 30, 50, 100]

sde = VPSDE()
score_model = None


def get_score_model():
    import lib.eps_ad.runners.diffpure_sde as diffpure_sde

    # 定义模型配置
    model_config = diffpure_sde.model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32,16,8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "1000",  # Modify this value to decrease the number of timesteps.
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
    print(f"model_config: {model_config}")

    # 根据模型配置创建模型并加载参数
    model, _ = diffpure_sde.create_model_and_diffusion(**model_config)
    model.load_state_dict(
        torch.load(f"models/256x256_diffusion_uncond.pt", map_location="cpu")
    )

    # 模型量化
    if model_config["use_fp16"]:
        model.convert_to_fp16()

    model = model.eval().to(device)
    return model


def extract_video_score(X, t_steps=diffuse_t):
    """计算单个视频的分数

    Args:
        X: 形状为 (num_frames, channels, height, width) 的张量,值域为 [0,1]
        t_steps: 扩散步数,默认为 50

    Returns:
        torch.Tensor: 形状为 (t_steps, num_frames, channels, height, width) 的张量,
            表示每个时间步的 Score
    """
    global score_model
    if score_model is None:
        score_model = get_score_model()

    # 检查输入值域
    assert (
        X.min() >= 0 and X.max() <= 1
    ), f"输入值域应该在 [0,1] 之间，但得到了：min={X.min()}, max={X.max()}"

    X = X.to(device)
    debug(f"X: {tensor_detail(X)}")

    # 将输入 X 的视频帧维度展平
    shape_t, shape_c, shape_h, shape_w = X.shape
    X = X.reshape(-1, shape_c, shape_h, shape_w)
    debug(f"X after reshape: {tensor_detail(X)}")

    # 将输入 X 的尺度从 [0, 1] 变换到 [-1, 1]
    X = 2 * X - 1
    debug(f"X after scale: {tensor_detail(X)}")

    with torch.no_grad():
        scores_at_t = []

        # t=1,2,5,10,15,20,30,50,100
        for t_step in range(1, t_steps + 1):

            # 时间步 t
            _t = torch.tensor(t_step / 1000, device=X.device)  # t/1000
            _t_expand = _t.expand(X.shape[0])  # 对张量 _t 进行广播 [batch_size]
            debug(f"_t: {tensor_detail(_t)}")
            debug(f"_t_expand: {tensor_detail(_t_expand)}")

            # 根据时间步 t 计算该时间步噪声扩散后的均值和标准差
            x_mean_at_t_step, x_std_at_t_step = sde.marginal_prob(X, _t_expand)
            debug(f"x_mean_at_t_step: {tensor_detail(x_mean_at_t_step)}")
            debug(f"x_std_at_t_step: {tensor_detail(x_std_at_t_step)}")

            # 引入一个高斯噪声
            z = torch.randn_like(X, device=X.device)
            debug(f"z: {tensor_detail(z)}")

            # 形成时间步 t 时的扰动样本
            perturbed_data = x_mean_at_t_step + x_std_at_t_step[:, None, None, None] * z
            debug(f"perturbed_data: {tensor_detail(perturbed_data)}")

            # 输入模型计算时间步 t 时的扰动样本的 Score
            _out = score_model(perturbed_data, _t_expand * 999)
            debug(f"_out: {tensor_detail(_out)}")

            # 对模型输出除以 sigma
            _, sigma = sde.marginal_prob(torch.zeros_like(X), _t_expand)
            debug(f"sigma: {tensor_detail(sigma)}")
            score = -_out / sigma[:, None, None, None]

            # Diffusion 模型的输出有两个部分，第一部分是 Score，第二部分不用管
            score, _ = torch.split(score, score.shape[1] // 2, dim=1)
            debug(f"score: {tensor_detail(score)}")

            # 检查张量形状
            assert (
                score.shape == X.shape
            ), f"Score 形状与输入不匹配：期望 {X.shape}，但得到了 {score.shape}"

            # 将维度恢复到原始形状
            score = score.reshape(shape_t, shape_c, shape_h, shape_w)
            debug(f"score after reshape: {tensor_detail(score)}")

            scores_at_t.append(score.detach())

        scores_at_t = torch.stack(scores_at_t)
        debug(f"scores_at_t: {tensor_detail(scores_at_t)}")  # (t_steps, t, c, h, w)

    return scores_at_t


def calc_video_score_by_timestep(scores_at_t):
    """计算不同时间步的视频分数

    Args:
        scores_at_t: 形状为 (t_steps, num_frames, channels, height, width) 的张量,
            表示每个时间步的 Score

    Returns:
        dict: key 为时间步,value 为对应时间步的标量分数
    """
    # 在时间维度上相邻的帧的 Score 相减
    score_diff = scores_at_t[:, 1:] - scores_at_t[:, :-1]  # (t_steps, t-1, c, h, w)
    debug(f"score_diff: {tensor_detail(score_diff)}")

    # 对前 selected_t_steps 分别求平均
    avg_scores_by_timestep = {}
    for by_t_step in selected_t_steps:
        if by_t_step > diffuse_t:
            break

        debug(f"by_t_step: {by_t_step}")

        # 取出当前时间步的 score_diff
        score = score_diff[:by_t_step]  # (t_steps, t-1, c, h, w)
        debug(f"score: {tensor_detail(score)}")

        score_mean = torch.mean(score, dim=0)  # (t-1, c, h, w)
        debug(f"score_mean: {tensor_detail(score_mean)}")

        # 计算 L2 范数
        score_mean_l2 = torch.norm(
            score_mean.view(score_mean.shape[0], -1), dim=1
        )  # (t-1,)
        debug(f"score_mean_l2: {tensor_detail(score_mean_l2)}")

        # 计算时间维度的平均
        score_mean_l2_mean = torch.mean(score_mean_l2)  # 标量
        debug(f"score_mean_l2_mean: {tensor_detail(score_mean_l2_mean)}")

        avg_scores_by_timestep[by_t_step] = score_mean_l2_mean

    return avg_scores_by_timestep


def calc_video_score_by_timestep_v2(scores_at_t):
    """计算不同时间步的视频分数

    Args:
        scores_at_t: 形状为 (t_steps, num_frames, channels, height, width) 的张量,
            表示每个时间步的 Score

    Returns:
        dict: key 为时间步,value 为对应时间步的标量分数
    """
    # 对前 selected_t_steps 分别求平均
    avg_scores_by_timestep = {}
    for by_t_step in selected_t_steps:
        if by_t_step > diffuse_t:
            break

        debug(f"by_t_step: {by_t_step}")

        # 取出当前时间步的分数
        score = scores_at_t[:by_t_step]  # (t_steps, t, c, h, w)
        debug(f"score: {tensor_detail(score)}")

        score_mean = torch.mean(score, dim=0)  # (t, c, h, w)
        debug(f"score_mean: {tensor_detail(score_mean)}")

        # 在时间维度上相邻的帧的 Score 相减
        score_diff = score_mean[1:] - score_mean[:-1]  # (t-1, c, h, w)
        debug(f"score_diff: {tensor_detail(score_diff)}")

        # 计算 L2 范数
        score_mean_l2 = torch.norm(
            score_diff.view(score_diff.shape[0], -1), dim=1
        )  # (t-1,)
        debug(f"score_mean_l2: {tensor_detail(score_mean_l2)}")

        # 计算时间维度的平均
        score_mean_l2_mean = torch.mean(score_mean_l2)  # 标量
        debug(f"score_mean_l2_mean: {tensor_detail(score_mean_l2_mean)}")

        avg_scores_by_timestep[by_t_step] = score_mean_l2_mean

    return avg_scores_by_timestep


def calc_video_score_by_timestep_v3(scores_at_t):
    """计算不同时间步的视频分数

    Args:
        scores_at_t: 形状为 (t_steps, num_frames, channels, height, width) 的张量,
            表示每个时间步的 Score

    Returns:
        dict: key 为时间步,value 为对应时间步的标量分数
    """
    # 对前 selected_t_steps 分别求平均
    avg_scores_by_timestep = {}
    for by_t_step in selected_t_steps:
        if by_t_step > diffuse_t:
            break

        debug(f"by_t_step: {by_t_step}")

        # 取出当前时间步的分数
        score = scores_at_t[:by_t_step]  # (t_steps, t, c, h, w)
        debug(f"score: {tensor_detail(score)}")

        score_mean = torch.mean(score, dim=0)  # (t, c, h, w)
        debug(f"score_mean: {tensor_detail(score_mean)}")

        # 计算 L2 范数
        score_mean_l2 = torch.norm(
            score_mean.view(score_mean.shape[0], -1), dim=1
        )  # (t,)
        debug(f"score_mean_l2: {tensor_detail(score_mean_l2)}")

        # 计算时间维度的平均
        score_mean_l2_mean = torch.mean(score_mean_l2)  # 标量
        debug(f"score_mean_l2_mean: {tensor_detail(score_mean_l2_mean)}")

        avg_scores_by_timestep[by_t_step] = score_mean_l2_mean

    return avg_scores_by_timestep


def calc_video_score_batch(dataloader):
    """计算视频的分数

    Args:
        dataloader: 数据加载器,每个批次返回 (videos, labels)
            videos: 形状为 (batch_size, num_frames, channels, height, width) 的张量,值域为 [0,1]
            labels: 形状为 (batch_size,) 的张量

    Returns:
        list: 每个批次的结果列表,每个元素为 (scores_dict, label)
            scores_dict: 字典,key 为时间步,value 为形状为 (batch_size,) 的分数张量
            label: 形状为 (batch_size,) 的标签张量
    """
    global score_model
    if score_model is None:
        score_model = get_score_model()

    batch_results = []

    # 这里输入尺度必须是 [0, 1]
    for idx, (X, labels) in enumerate(dataloader):
        X = X.to(device)
        debug(f"X: {tensor_detail(X)}")

        # 将输入 X 的批量维度和视频帧维度（前两个维度）合并
        shape_b, shape_t, shape_c, shape_h, shape_w = X.shape
        X = X.reshape(shape_b * shape_t, shape_c, shape_h, shape_w)
        debug(f"X after reshape: {tensor_detail(X)}")

        # 将输入 X 的尺度从 [0, 1] 变换到 [-1, 1]
        X = 2 * X - 1
        debug(f"X: {tensor_detail(X)}")

        with torch.no_grad():
            scores_at_t = []

            # t=1,2,5,10,15,20,30,50,100
            for by_t_step in range(1, diffuse_t + 1):

                # 时间步 t
                _t = torch.tensor(by_t_step / 1000, device=X.device)  # t/1000
                _t_expand = _t.expand(X.shape[0])  # 对张量 _t 进行广播 [batch_size]
                debug(f"_t: {tensor_detail(_t)}")
                debug(f"_t_expand: {tensor_detail(_t_expand)}")

                # 根据时间步 t 计算该时间步噪声扩散后的均值和标准差
                x_mean_at_t_step, x_std_at_t_step = sde.marginal_prob(X, _t_expand)
                debug(f"x_mean_at_t_step: {tensor_detail(x_mean_at_t_step)}")
                debug(f"x_std_at_t_step: {tensor_detail(x_std_at_t_step)}")

                # 引入一个高斯噪声
                z = torch.randn_like(X, device=X.device)
                debug(f"z: {tensor_detail(z)}")

                # 形成时间步 t 时的扰动样本
                perturbed_data = (
                    x_mean_at_t_step + x_std_at_t_step[:, None, None, None] * z
                )
                debug(f"perturbed_data: {tensor_detail(perturbed_data)}")

                # 输入模型计算时间步 t 时的扰动样本的 Score
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                _out = score_model(perturbed_data, _t_expand * 999)
                debug(f"_out: {tensor_detail(_out)}")

                # 对模型输出除以 sigma
                _, sigma = sde.marginal_prob(torch.zeros_like(X), _t_expand)
                debug(f"sigma: {tensor_detail(sigma)}")
                score = -_out / sigma[:, None, None, None]

                # Diffusion 模型的输出有两个部分，第一部分是 Score，第二部分不用管
                score, _ = torch.split(score, score.shape[1] // 2, dim=1)
                debug(f"score: {tensor_detail(score)}")

                # 检查张量形状
                assert (
                    score.shape == X.shape
                ), f"Score 形状与输入不匹配：期望 {X.shape}，但得到了 {score.shape}"

                # 将维度恢复到原始形状
                score = score.reshape(shape_b, shape_t, shape_c, shape_h, shape_w)
                debug(f"score after reshape: {tensor_detail(score)}")

                # 在时间维度上相邻的帧的 Score 相减
                score_diff = score[:, 1:] - score[:, :-1]
                debug(f"score_diff: {tensor_detail(score_diff)}")  # (b, t-1, c, h, w)

                scores_at_t.append(score_diff.detach())

        scores_at_t = torch.stack(scores_at_t)
        debug(
            f"scores_at_t: {tensor_detail(scores_at_t)}"
        )  # (t_steps, b, t-1, c, h, w)

        # 对前 selected_t_steps 分别求平均
        batch_avg_scores = []  # 存储每个样本的 avg_scores_by_timestep

        for b in range(shape_b):  # 遍历每个样本

            avg_scores_by_timestep = {}
            for by_t_step in selected_t_steps:

                if by_t_step > diffuse_t:
                    break

                debug(f"by_t_step: {by_t_step}")

                # 取出当前样本的 score_diff
                score_diff = scores_at_t[:by_t_step, b]  # (t_steps, t-1, c, h, w)
                debug(f"score_diff: {tensor_detail(score_diff)}")

                score_diff_mean = torch.mean(score_diff, dim=0)  # (t-1, c, h, w)
                debug(f"score_diff_mean: {tensor_detail(score_diff_mean)}")

                # 计算 L2 范数
                score_diff_mean_l2 = torch.norm(
                    score_diff_mean.view(shape_t - 1, -1), dim=1
                )  # (t-1,)
                debug(f"score_diff_mean_l2: {tensor_detail(score_diff_mean_l2)}")

                # 计算时间维度的平均
                score_diff_mean_l2_mean = torch.mean(score_diff_mean_l2)  # 标量
                debug(
                    f"score_diff_mean_l2_mean: {tensor_detail(score_diff_mean_l2_mean)}"
                )

                avg_scores_by_timestep[by_t_step] = score_diff_mean_l2_mean

            batch_avg_scores.append(avg_scores_by_timestep)

        # 将当前批次的结果添加到列表中,每个样本对应其标签
        for b in range(shape_b):
            batch_results.append((batch_avg_scores[b], labels[b]))

    debug(f"batch_results: {batch_results}")
    return batch_results


if __name__ == "__main__":

    # 设置随机种子
    set_random_seed(1235)

    # 创建数据集和数据加载器
    class VideoDataset(Dataset):
        def __init__(self, size=50, img_size=(10, 3, 256, 256)):
            self.size = size
            self.img_size = img_size
            self.data = torch.rand(size, *img_size)  # 生成 0-1 之间的随机图像数据
            self.labels = torch.randint(0, 2, (size,))  # 生成随机 0/1 标签

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # 创建数据集实例
    dataset = VideoDataset()

    # 创建数据加载器
    batch_size = 1
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 从数据集中取一个样本进行测试
    test_video, test_label = dataset[0]  # 获取第一个样本
    print(f"测试视频形状: {test_video.shape}")
    print(f"测试标签: {test_label}")

    scores_at_t = extract_video_score(test_video)
    print(f"scores_at_t: {tensor_detail(scores_at_t)}")

    # 测试 calc_video_score_by_timestep 函数
    print("\n测试 calc_video_score_by_timestep 函数:")
    scores_by_timestep = calc_video_score_by_timestep(scores_at_t)
    for t_step, score in scores_by_timestep.items():
        print(f"t={t_step}: {tensor_detail(score)}")
