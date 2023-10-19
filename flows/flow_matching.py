from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
from absl import logging

_RTOL = 1e-5
_ATOL = 1e-5


class CNF(nn.Module):
    def __init__(
        self,
        net,
    ):
        super().__init__()
        self.net = net

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor,
        **kwargs,
    ) -> Tensor:
        if t.numel() == 1:
            t = t.expand(x.size(0))
        _pred, inters = self.net(x, t, y, **kwargs)

        return _pred

    # @torch.cuda.amp.autocast()
    def training_losses(self, x, y, sigma_min, **kwargs):
        noise = torch.randn_like(x)

        t = torch.rand(len(x), device=x.device, dtype=x.dtype)
        t_ = t[:, None, None, None]  # [B, 1, 1, 1]
        x_new = t_ * x + (1 - (1 - sigma_min) * t_) * noise
        u = x - (1 - sigma_min) * noise

        return (
            (self.forward(t, x_new, y=y, **kwargs) - u)  # self.forward = vector_field
            .square()
            .mean(dim=(1, 2, 3))
        )

    def encode(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        # if y is not None:
        func = lambda t, x: self(t, x, y=y, **kwargs)

        ode_kwargs = dict(
            method="dopri5",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
        )

        return odeint(
            func,
            x,
            # 0.0,
            torch.tensor([1.0, 0.0], device=x.device, dtype=x.dtype),
            # phi=self.parameters(),
            **ode_kwargs,
        )[-1]

    def decode(
        self,
        z: Tensor,
        y: Tensor,
        **kwargs,
    ) -> Tensor:
        func = lambda t, x: self(t, x, y=y, **kwargs)

        ode_kwargs = dict(
            method="euler",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
            options=dict(step_size=0.01),
        )

        return odeint(
            func,
            z,
            # 0.0,
            torch.tensor([0.0, 1.0], device=z.device, dtype=z.dtype),
            # phi=self.parameters(),
            **ode_kwargs,
        )[-1]

    @torch.no_grad()
    def sample_euler_raw(
        self, z, step_num, return_x_est=False, return_x_est_num=None, **kwargs
    ):
        dt = 1.0 / step_num
        traj = []  # to store the trajectory

        z = z.detach().clone()
        bs = len(z)

        est = []

        if return_x_est:
            est_ids = [
                int(i * step_num / return_x_est_num) for i in range(return_x_est_num)
            ]

        traj.append(z.detach().clone())
        for i in range(0, step_num, 1):
            t = torch.ones(bs, device=z.device) * i / step_num
            pred = self.forward(z, t, **kwargs)

            _est_now = z + (1 - i * 1.0 / step_num) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]
