import os
import random
import numpy as np

import torch
import torchsde
import torchvision.utils as tvu
from eps_ad.runners.diffpure_sde import RevVPSDE
from eps_ad.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state["optimizer"].load_state_dict(loaded_state["optimizer"])
    state["model"].load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]


class RevGuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # load model
        img_shape = (3, 256, 256)
        model_dir = "pretrained/guided_diffusion"
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        print(f"model_config: {model_config}")
        model, _ = create_model_and_diffusion(**model_config)
        model.load_state_dict(
            torch.load(f"{model_dir}/256x256_diffusion_uncond.pt", map_location="cpu")
        )

        if model_config["use_fp16"]:
            model.convert_to_fp16()

        model.eval().to(self.device)

        self.model = model
        self.rev_vpsde = RevVPSDE(
            model=model,
            score_type=args.score_type,
            img_shape=img_shape,
            model_kwargs=None,
        ).to(self.device)
        self.betas = self.rev_vpsde.discrete_betas.float().to(self.device)

        print(f"t: {args.t}, rand_t: {args.rand_t}, t_delta: {args.t_delta}")
        print(f"use_bm: {args.use_bm}")

    def image_editing_sample(self, img, bs_id=0, tag=None, t_size=2):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))  # c*h*w

        if tag is None:
            tag = "rnd" + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, "bs" + str(bs_id) + "_" + tag)

        assert img.ndim == 4, img.ndim
        img = img.to(self.device)
        x0 = img

        if bs_id < 2:
            os.makedirs(out_dir, exist_ok=True)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f"original_input.png"))

        xs = []
        for it in range(self.args.sample_step):
            # sample_step controls different variances of every diffusion
            e = torch.randn_like(x0).to(self.device)
            total_noise_levels = self.args.t  # select the noise scales in [0,t^star]
            if self.args.rand_t:
                total_noise_levels = self.args.t + np.random.randint(
                    -self.args.t_delta, self.args.t_delta
                )
                print(f"total_noise_levels: {total_noise_levels}")
            a = (
                (1 - self.betas).cumprod(dim=0).to(self.device)
            )  # self.betas [0,1] \in R^[1000]
            x = (
                x0 * a[total_noise_levels - 1].sqrt()
                + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            )  # diffuse image at first [0, t=t^{star}]
            if self.args.detection_flag:
                x = x0
            if bs_id < 2:
                if not self.args.detection_flag:
                    tvu.save_image(
                        (x + 1) * 0.5, os.path.join(out_dir, f"init_{it}.png")
                    )

            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0, t1 = (
                1 - self.args.t * 1.0 / 1000 + epsilon_dt0,
                1 - epsilon_dt1,
            )  # purify from [1-t*, 1], note they are symmetric
            # t_size = 3
            ts = torch.linspace(t0, t1, t_size).to(self.device)

            x_ = x.view(batch_size, -1)  # (batch_size, state_size)
            if self.args.use_bm:
                bm = torchsde.BrownianInterval(
                    t0=t0, t1=t1, size=(batch_size, state_size), device=self.device
                )
                xs_ = torchsde.sdeint_adjoint(
                    self.rev_vpsde, x_, ts, method="euler", bm=bm
                )
            else:
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method="euler")
            x0 = xs_[-1].view(x.shape)  # (batch_size, c, h, w)

            if bs_id < 2:
                if not self.args.detection_flag:
                    torch.save(x0, os.path.join(out_dir, f"samples_{it}.pth"))
                    tvu.save_image(
                        (x0 + 1) * 0.5, os.path.join(out_dir, f"samples_{it}.png")
                    )
            if self.args.detection_flag:
                x0 = xs_.view(-1, x.shape[1], x.shape[2], x.shape[3])
                if bs_id < 2:
                    tvu.save_image(
                        (x0 + 1) * 0.5, os.path.join(out_dir, f"samples_{it}.png")
                    )
                ts_cat = (
                    (1 - ts)
                    .view(t_size, 1)
                    .expand(t_size, x.shape[0])
                    .contiguous()
                    .view(-1)
                )
            xs.append(x0)

        return (
            torch.cat(xs, dim=0)
            if not self.args.detection_flag
            else torch.cat(xs, dim=0)
        ), ts_cat
