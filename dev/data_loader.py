import numpy as np
import torch
from config import Config
from model import BipartiteData

def load_bipartite_data(cfg: Config) -> BipartiteData:
    """
    Load bipartite graph data for fibers and classes.
    """
    class_info = torch.tensor(
        np.loadtxt(cfg.data_file),
        dtype=torch.float,
        device=cfg.device
    )
    x_t = class_info
    x_s = torch.arange(
        cfg.n_fibers, dtype=torch.float,
        device=cfg.device
    ).unsqueeze(-1)

    edge_index = torch.cartesian_prod(
        torch.arange(cfg.n_fibers, device=cfg.device),
        torch.arange(cfg.n_classes, device=cfg.device)
    ).T

    edge_attr = torch.rand(
        (edge_index.size(1), cfg.fdim),
        device=cfg.device
    )
    x_u = torch.zeros((1, cfg.fdim), device=cfg.device)

    return BipartiteData(edge_index, x_s, x_t, edge_attr, x_u)
