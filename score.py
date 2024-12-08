import argparse
import yaml
import os
import time

import random
import numpy as np
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

import eps_ad.utils as utils
from eps_ad.utils import str2bool, get_image_classifier
from eps_ad.runners.diffpure_sde import RevGuidedDiffusion

# from eps_ad.score_sde import sde_lib
from lib.sde import VPSDE

# from eps_ad.score_sde.models import utils as mutils
# from eps_ad.runners.diffpure_sde import RevVPSDE


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train = False
continuous = True

sde = VPSDE()

model = SDE_Adv_Model(args, config)
model = model.eval().to(device)


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        self.detection_flag = True

        # self.classifier = get_image_classifier(args.classifier_name).to(device)

        self.runner = RevGuidedDiffusion(args, config, device=device)

        self.register_buffer("counter", torch.zeros(1, device=device))
        """
        register_buffer 可以确保 counter 总是和模型在同一个设备上（CPU 或 GPU）。如果直接用属性，在模型迁移设备时需要手动处理这个变量。
        register_buffer 注册的变量会被视为模型状态的一部分。在保存和加载模型时（state_dict()），这些 buffer 会自动被保存和恢复；普通属性则不会被自动包含在模型状态中。
        虽然对于简单的计数器来说，这种差异可能不是特别明显，但使用 register_buffer 是更规范的做法。
        """
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        """
        if not self.detection_flag:
            # 防御模式：净化图像并进行分类
            x_re = self.runner.image_editing_sample(...)
            out = self.classifier((x_re + 1) * 0.5)
        else:
            # 检测模式：返回净化后的图像和时间序列信息
            x_re, ts_cat = self.runner.image_editing_sample(...)
        """

        counter = self.counter.item()
        if counter % 5 == 0:
            print(f"diffusion times: {counter}")

        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        start_time = time.time()
        if not self.detection_flag:
            x_re = self.runner.image_editing_sample(
                (x - 0.5) * 2, bs_id=counter, tag=self.tag
            )  # diffusion+purify
        else:
            x_re, ts_cat = self.runner.image_editing_sample(
                (x - 0.5) * 2, bs_id=counter, tag=self.tag, t_size=self.args.t_size
            )  # diffusion+purify

        minutes, seconds = divmod(time.time() - start_time, 60)

        if not self.detection_flag:

            x_re = F.interpolate(
                x_re, size=(224, 224), mode="bilinear", align_corners=False
            )

            if counter % 5 == 0:
                print(f"x shape (before diffusion models): {x.shape}")
                print(f"x shape (before classifier): {x_re.shape}")
                print(
                    "Sampling time per batch: {:0>2}:{:05.2f}".format(
                        int(minutes), seconds
                    )
                )

            out = self.classifier((x_re + 1) * 0.5)

        self.counter += 1

        return out if not self.detection_flag else x_re, ts_cat


def score_fn(X, T):  # 加完噪声的 X 和 时间 T
    """Compute the output of the score-based model.

    Args:
    model: The score model.
    x: A mini-batch of input data.
    labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
    """

    labels = T * 999
    _out = model(X, labels)

    # 通过标准差和翻转符号缩放神经网络输出
    std = sde.marginal_prob(torch.zeros_like(X), T)[1]
    score = -_out / std[:, None, None, None]
    return score


def detection_test_ensattack(args, config):

    score_adv_list = []

    diffuse_t = 100

    # 这里输入尺度必须是 [0, 1]
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            for t in range(1, diffuse_t + 1):

                curr_t = torch.tensor(t / 1000, device=x.device)

                # 噪声
                z = torch.randn_like(x, device=x.device)
                x_mean, x_std = sde.marginal_prob(
                    2 * x - 1,  # 这里输入尺度变换成 [-1, 1]
                    curr_t.expand(x.shape[0]),  # 对张量进行广播
                )
                perturbed_data = x_mean + x_std[:, None, None, None] * z
                score = score_fn(perturbed_data, curr_t.expand(x.shape[0]))
                # Diffusion 模型的输出有两个部分，第一部分是 Score，第二部分不用管
                score, _ = torch.split(score, score.shape[1] // 2, dim=1)
                # 确保 Score 形状与输入相同
                assert score.shape == x.shape, f"{x.shape}, {score.shape}"

                score_adv_list.append(score.detach())


from eps_ad.score_sde import sde_lib
from eps_ad.score_sde.models import utils as mutils
from eps_ad.runners.diffpure_sde import RevVPSDE


def get_score(x, y, t):
    x = x.to(device)
    y = y.to(device)
    t = torch.tensor(t / 1000, device=device)

    z = torch.randn_like(x, device=x.device)
    x_mean, x_std = sde.marginal_prob(2 * x - 1, t.expand(x.shape[0]))
    perturbed_data = x_mean + x_std[:, None, None, None] * z
    score = score_fn(perturbed_data, t.expand(x.shape[0]))


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    # diffusion models
    parser.add_argument(
        "--config",
        type=str,
        # default="cifar10.yml",
        default="imagenet.yml",
        help="Path to the config file",
    )
    parser.add_argument("--data_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--seed", type=int, default=1235, help="Random seed")
    parser.add_argument(
        "--exp",
        type=str,
        default="./exp_results",
        help="Path for saving running related data.",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        # default="cifar10",
        default="imagenet",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_step", type=int, default=1, help="Total sampling steps"
    )
    parser.add_argument("--t", type=int, default=1000, help="Sampling noise scale")
    parser.add_argument(
        "--t_delta",
        type=int,
        default=15,
        help="Perturbation range of sampling noise scale",
    )
    parser.add_argument(
        "--rand_t",
        type=str2bool,
        default=False,
        help="Decide if randomize sampling noise scale",
    )
    parser.add_argument("--diffusion_type", type=str, default="sde", help="[ddpm, sde]")
    parser.add_argument(
        "--score_type",
        type=str,
        default="score_sde",
        help="[guided_diffusion, score_sde]",
    )
    parser.add_argument(
        "--eot_iter", type=int, default=20, help="only for rand version of autoattack"
    )
    parser.add_argument(
        "--use_bm", action="store_true", help="whether to use brownian motion"
    )
    parser.add_argument("--datapath", type=str, default="./dataset")

    # Detection
    parser.add_argument("--clean_score_flag", action="store_true")
    parser.add_argument(
        "--detection_datapath", type=str, default="./score_diffusion_t_cifar"
    )  # ./score_diffusion_t_cifar
    # parser.add_argument('--detection_flag', action='store_true')
    # parser.add_argument('--detection_ensattack_flag', action='store_true')
    parser.add_argument("--detection_ensattack_norm_flag", action="store_true")
    parser.add_argument("--generate_1w_flag", action="store_true")
    parser.add_argument("--single_vector_norm_flag", action="store_true")
    parser.add_argument("--t_size", type=int, default=10)
    parser.add_argument("--diffuse_t", type=int, default=100)
    parser.add_argument("--perb_image", action="store_true")

    # LDSDE
    parser.add_argument("--sigma2", type=float, default=1e-3, help="LDSDE sigma2")
    parser.add_argument("--lambda_ld", type=float, default=1e-2, help="lambda_ld")
    parser.add_argument("--eta", type=float, default=5.0, help="LDSDE eta")
    parser.add_argument(
        "--step_size", type=float, default=1e-2, help="step size for ODE Euler method"
    )

    # adv
    parser.add_argument(
        "--domain",
        type=str,
        # default="cifar10",
        default="imagenet",
        help="which domain: celebahq, cat, car, imagenet",
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="cifar10-wideresnet-28-10",
        help="which classifier to use",
    )
    parser.add_argument("--partition", type=str, default="val")
    parser.add_argument("--adv_batch_size", type=int, default=64)
    parser.add_argument("--attack_type", type=str, default="square")
    parser.add_argument("--lp_norm", type=str, default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--attack_version", type=str, default="standard")

    # additional attack settings
    parser.add_argument(
        "--num-steps", default=5, type=int, help="perturb number of steps"
    )
    parser.add_argument("--random", default=True, help="random initialization for PGD")
    parser.add_argument(
        "--attack_methods",
        type=str,
        nargs="+",
        default=[
            "FGSM",
            "PGD",
            "BIM",
            "MIM",
            "TIM",
            "CW",
            "DI_MIM",
            "FGSM_L2",
            "PGD_L2",
            "BIM_L2",
            "MM_Attack",
            "AA_Attack",
        ],
    )
    parser.add_argument("--mim_momentum", default=1.0, type=float, help="mim_momentum")
    parser.add_argument(
        "--epsilon", default=0.01568, type=float, help="perturbation"
    )  # 0.01568, type=float,help='perturbation')

    parser.add_argument("--num_sub", type=int, default=64, help="imagenet subset")
    parser.add_argument("--adv_eps", type=float, default=0.031373, help="0.031373")
    parser.add_argument("--gpu_ids", type=str, default="3,4")

    # vmi-fgsm
    parser.add_argument(
        "--momentum", default=1.0, type=float, help="momentum of the attack"
    )
    parser.add_argument(
        "--number",
        default=20,
        type=int,
        help="the number of images for variance tuning",
    )
    parser.add_argument(
        "--beta", default=1.5, type=float, help="the bound for variance tuning"
    )
    parser.add_argument(
        "--prob", default=0.5, type=float, help="probability of using diverse inputs"
    )
    parser.add_argument(
        "--image_resize", default=331, type=int, help="heigth of each input image"
    )

    args = parser.parse_args()
    args.step_size_adv = args.epsilon / args.num_steps

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # parse config file
    with open(os.path.join("eps_ad/configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == "__main__":
    args, config = parse_args_and_config()
    # robustness_eval(args, config)
