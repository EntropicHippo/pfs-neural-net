import torch
import numpy as np
from torch_scatter import scatter
from typing import Tuple
from config import Config
from model import BipartiteData, MPNN

def soft_floor(x: torch.Tensor,
              sharpness: float,
              noise_level: float = 0.3) -> torch.Tensor:
    """
    Differentiable soft floor function.
    """
    noise = noise_level * (torch.rand_like(x) - 0.5)
    x_noisy = x + noise
    r = torch.exp(-1.0 / sharpness) if sharpness != 0 else torch.tensor(0.0, device=x.device)
    pi = torch.tensor(np.pi, device=x.device)
    num = r * torch.sin(2 * pi * x_noisy)
    den = 1 - r * torch.cos(2 * pi * x_noisy)
    return x_noisy + (torch.atan(num/den) - torch.atan(r/(1-r))) / pi

def compute_loss(model: MPNN,
                 data: BipartiteData,
                 cfg: Config,
                 sharpness: float,
                 final_output: bool = False
                 ) -> Tuple:
    """
    Compute training loss and metrics.
    """
    src, tgt = data.edge_index
    class_info = data.x_t
    t_i = class_info[:, 0]
    n_i = class_info[:, 1] / cfg.n_fields

    edge_time = model.predict_edge_time(
        data.edge_attr,
        scale=cfg.total_time / cfg.n_classes
    )
    visited = edge_time / t_i.repeat(cfg.n_fibers)

    galaxies = soft_floor(visited, sharpness)
    galaxies = torch.clamp(galaxies, min=0.0)
    n_prime = scatter(
        galaxies, tgt, dim=0,
        dim_size=cfg.n_classes, reduce='sum'
    )

    time_spent = galaxies * t_i.repeat(cfg.n_fibers)
    completeness = n_prime / n_i
    utility = completeness.min()

    class_over = torch.relu(n_prime - n_i)
    class_penalty = cfg.p_class * (class_over ** 2).sum()

    fiber_time = scatter(
        time_spent, src, dim=0,
        dim_size=cfg.n_fibers, reduce='sum'
    )
    fiber_over = torch.relu(fiber_time - cfg.total_time)
    fiber_penalty = cfg.p_fiber * (fiber_over ** 2).sum()

    time_matrix = time_spent.view(cfg.n_fibers, cfg.n_classes)
    var_term = time_matrix.var(dim=0).sum()

    loss = -cfg.w_utils * utility + class_penalty + fiber_penalty - cfg.w_var * var_term

    if final_output:
        return (loss, utility,
                completeness.detach().cpu().numpy(),
                n_prime.detach().cpu().numpy(),
                fiber_time.detach().cpu().numpy(),
                time_spent.detach().cpu().numpy(),
                var_term.detach().cpu().item())
    return loss, utility
