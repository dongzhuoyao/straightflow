import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
import wandb

# Implement some helper functions


def plot_trajectories(traj, fig_path):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(fig_path)
    return fig_path


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def sample_normal(n):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(2), torch.eye(2)
    )
    return m.sample((n,))


def log_normal_density(x):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(x.shape[-1]), torch.eye(x.shape[-1])
    )
    return m.log_prob(x)


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def log_8gaussian_density(x, scale=5, var=0.1):
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    centers = centers.T.reshape(1, 2, 8)
    # calculate shifted xs [batch, centers, dims]
    x = (x[:, :, None] - centers).mT
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(x.shape[-1]), math.sqrt(var) * torch.eye(x.shape[-1])
    )
    log_probs = m.log_prob(x)
    log_probs = torch.logsumexp(log_probs, -1)
    return log_probs


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)


sigma = 0.01
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
# FM = ConditionalFlowMatcher(sigma=sigma)


wandb_name = f"cfg_sigma{sigma}"
wandb.init(project="cfm", name=wandb_name)


def sample_xt(x0, x1, t, eps):
    mu_t = t * x1 + (1 - t) * x0

    sigma_t = pad_t_like_x(sigma, x0)
    return mu_t + sigma_t * eps


print("begin training")
start = time.time()
for k in range(20000):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)

    # t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
    t = torch.rand(len(x0), 1).type_as(x0)
    eps = torch.randn_like(x0)
    xt = sample_xt(x0, x1, t, eps)
    ut = x1 - x0

    vt = model(torch.cat([xt, t], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 500 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            fig_path = plot_trajectories(traj.cpu().numpy(), fig_path="trajectory.png")
            wandb.log({"loss": loss.item(), "trajectory": wandb.Image(fig_path)})
torch.save(model, f"{savedir}/cfm_v1.pt")
