import random
import numpy as np
import torch
from lib.sde import VPSDE
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sde = VPSDE()


def get_score_model():
    import eps_ad.runners.diffpure_sde as diffpure_sde

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


def calc_video_score(dataloader):
    score_adv_list = []
    diffuse_t = 100

    model = get_score_model()

    # 这里输入尺度必须是 [0, 1]
    for idx, (X, _) in enumerate(dataloader):
        X = X.to(device)

        # 将输入 X 的尺度从 [0, 1] 变换到 [-1, 1]
        X = 2 * X - 1

        with torch.no_grad():
            for t in range(1, diffuse_t + 1):

                # 时间步 t
                _t = torch.tensor(t / 1000, device=X.device)  # t/1000
                _t_expand = _t.expand(X.shape[0])  # 对张量 _t 进行广播 [batch_size]

                # 根据时间步 t 计算该时间步噪声扩散后的均值和标准差
                x_mean_at_t_step, x_std_at_t_step = sde.marginal_prob(X, _t_expand)

                # 引入一个高斯噪声
                z = torch.randn_like(X, device=X.device)

                # 形成时间步 t 时的扰动样本
                perturbed_data = (
                    x_mean_at_t_step + x_std_at_t_step[:, None, None, None] * z
                )

                # 输入模型计算时间步 t 时的扰动样本的 Score
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                _out = model(perturbed_data, _t_expand * 999)

                # 对模型输出除以 sigma
                _, sigma = sde.marginal_prob(torch.zeros_like(X), _t_expand)
                score = -_out / sigma[:, None, None, None]

                # Diffusion 模型的输出有两个部分，第一部分是 Score，第二部分不用管
                score, _ = torch.split(score, score.shape[1] // 2, dim=1)
                # 确保 Score 形状与输入相同
                assert score.shape == X.shape, f"{X.shape}, {score.shape}"

                score_adv_list.append(score.detach())


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # 设置 PyTorch 的 CPU 随机种子
    if torch.cuda.is_available():  # 如果 GPU 可用，设置 PyTorch 的 GPU 随机种子
        torch.cuda.manual_seed_all(random_seed)


if __name__ == "__main__":

    # 设置随机种子
    set_random_seed(random_seed=1235)

    # 优化 GPU 运算速度
    """当设置为 True 时，cuDNN 会自动寻找最适合当前硬件的卷积算法。
    系统会先花费一些时间找到最优算法，然后在接下来的运行中一直使用这个最优算法。
    第一次运行时会略微变慢（因为要寻找最优算法），之后的运行会明显加速，并会占用更多的显存。"""
    torch.backends.cudnn.benchmark = True

    # 创建数据集和数据加载器
    class SimpleDataset(Dataset):
        def __init__(self, size=1000, img_size=(3, 255, 255)):
            self.size = size
            self.img_size = img_size
            self.data = torch.randn(size, *img_size)  # 生成随机图像数据
            self.labels = torch.randint(0, 2, (size,))  # 生成随机 0/1 标签

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # 创建数据集实例
    dataset = SimpleDataset()

    # 创建数据加载器
    batch_size = 32
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    calc_video_score(dataloader)
