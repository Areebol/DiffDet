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


def get_score_model(args, config):
    model = RevGuidedDiffusion(args, config, device=device).model
    model = model.eval().to(device)
    return model


def detection_test_ensattack():
    score_adv_list = []
    diffuse_t = 100

    model = get_score_model(args, config)

    # 这里输入尺度必须是 [0, 1]
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # 将输入 X 的尺度从 [0, 1] 变换到 [-1, 1]
        x = 2 * x - 1

        with torch.no_grad():
            for t in range(1, diffuse_t + 1):

                # 时间步 t
                _t = torch.tensor(t / 1000, device=x.device)  # t/1000
                _t_expand = _t.expand(x.shape[0])  # 对张量 _t 进行广播 [batch_size]

                # 根据时间步 t 计算该时间步噪声扩散后的均值和标准差
                x_mean_at_t_step, x_std_at_t_step = sde.marginal_prob(x, _t_expand)

                # 引入一个高斯噪声
                z = torch.randn_like(x, device=x.device)

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
                _, sigma = sde.marginal_prob(torch.zeros_like(x), _t_expand)
                score = -_out / sigma[:, None, None, None]

                # Diffusion 模型的输出有两个部分，第一部分是 Score，第二部分不用管
                score, _ = torch.split(score, score.shape[1] // 2, dim=1)
                # 确保 Score 形状与输入相同
                assert score.shape == x.shape, f"{x.shape}, {score.shape}"

                score_adv_list.append(score.detach())


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
