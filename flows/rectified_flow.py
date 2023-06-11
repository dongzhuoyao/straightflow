from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
from absl import logging

from flows.sampling_rflow import get_rectified_flow_sampler

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
        if True:
            eps = 1e-3
            z0 = torch.randn_like(x)
            sde_T = 1.0

            t = torch.rand(len(x), device=x.device, dtype=x.dtype) * (sde_T - eps) + eps
            t_expand = t[:, None, None, None]

            perturbed_data = t_expand * x + (1.0 - t_expand) * z0
            target = x - z0
            score = self.forward(
                t, perturbed_data, y=y, **kwargs
            )  ### Copy from models/utils.py
            losses = torch.square(score - target)
            return losses.mean(dim=(1, 2, 3))
        else:
            t = torch.rand(len(x), device=x.device, dtype=x.dtype)
            t_ = t[:, None, None, None]  # [B, 1, 1, 1]
            x_new = t_ * x + (1 - (1 - sigma_min) * t_) * noise
            u = x - (1 - sigma_min) * noise

            return (
                (
                    self.forward(t, x_new, y=y, **kwargs) - u
                )  # self.forward = vector_field
                .square()
                .mean(dim=(1, 2, 3))
            )

    def encode(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError("Not carefully checked yet")

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
        class _SDE:
            def __init__(
                self,
            ):
                self.ode_tol = 1e-5
                self.T = 1

        def _inverse_scaler(x):
            return x

        _sampler_fn = get_rectified_flow_sampler(
            _SDE(), shape=z.shape, inverse_scaler=_inverse_scaler, device=z.device
        )

        func = lambda t, x: self(t, x, y=y, **kwargs)

        result, _ = _sampler_fn(func, z)
        return result

    def decode_from_fm(
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
