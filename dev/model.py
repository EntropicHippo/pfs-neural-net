import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
from config import Config

__all__ = ["BipartiteData", "GNN"]

def _make_message_embed(x_s: Tensor,
                        x_t: Tensor,
                        edge_attr: Tensor,
                        u: Tensor,
                        edge_index: Tensor) -> Tensor:
    """
    Prepare concatenated message embedding for edges.
    """
    src, tgt = edge_index
    E = edge_attr.size(0)
    global_feat = u.expand(E, -1)
    return torch.cat([x_s[src], x_t[tgt], edge_attr, global_feat], dim=-1)


class BipartiteData(Data):
    """
    Data for bipartite graph: fibers (sources) to classes (targets).
    """
    def __init__(self,
                 edge_index: Tensor,
                 x_s: Tensor,
                 x_t: Tensor,
                 edge_attr: Tensor,
                 x_u: Tensor):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.edge_attr = edge_attr
        self.x_u = x_u

    def to(self, device: torch.device) -> "BipartiteData":
        return BipartiteData(
            self.edge_index.to(device),
            self.x_s.to(device),
            self.x_t.to(device),
            self.edge_attr.to(device),
            self.x_u.to(device)
        )


class EdgeModel(torch.nn.Module):
    """
    Edge update MLP.
    """
    def __init__(self, fdim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * fdim, 4 * fdim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * fdim, fdim),
            torch.nn.BatchNorm1d(fdim)
        )

    def forward(self,
                x_s: Tensor,
                x_t: Tensor,
                edge_attr: Tensor,
                u: Tensor,
                edge_index: Tensor) -> Tensor:
        h = _make_message_embed(x_s, x_t, edge_attr, u, edge_index)
        return self.mlp(h)


class NodeModel(torch.nn.Module):
    """
    Node update MLP with aggregation.
    """
    def __init__(self, fdim: int, aggregator: str = 'mean'):
        super().__init__()
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * fdim, 2 * fdim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * fdim, fdim)
        )
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * fdim, fdim),
            torch.nn.ReLU(),
            torch.nn.Linear(fdim, fdim),
            torch.nn.BatchNorm1d(fdim)
        )
        self.aggregator = aggregator

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_attr: Tensor,
                is_source: bool = True) -> Tensor:
        src, tgt = edge_index
        if is_source:
            msg = torch.cat([x[tgt], edge_attr], dim=-1)
            msg = self.message_mlp(msg)
            agg = scatter(msg, src, dim=0, reduce=self.aggregator, dim_size=x.size(0))
        else:
            msg = torch.cat([x[src], edge_attr], dim=-1)
            msg = self.message_mlp(msg)
            agg = scatter(msg, tgt, dim=0, reduce='sum', dim_size=x.size(0))
        return self.update_mlp(torch.cat([x, agg], dim=-1))


class GlobalModel(torch.nn.Module):
    """
    Global feature update MLP.
    """
    def __init__(self, fdim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * fdim, 3 * fdim),
            torch.nn.ReLU(),
            torch.nn.Linear(3 * fdim, fdim),
            torch.nn.LayerNorm(fdim)
        )

    def forward(self,
                x_s: Tensor,
                x_t: Tensor,
                x_u: Tensor) -> Tensor:
        s_mean = x_s.mean(dim=0, keepdim=True)
        t_mean = x_t.mean(dim=0, keepdim=True)
        h = torch.cat([x_u, s_mean, t_mean], dim=-1)
        return self.mlp(h)


class GNN(torch.nn.Module):
    """
    Graph Neural Network for bipartite allocation.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.fdim = cfg.fdim
        self.encoder_s = torch.nn.Sequential(
            torch.nn.Linear(1, self.fdim),
            torch.nn.ReLU()
        )
        self.encoder_t = torch.nn.Sequential(
            torch.nn.Linear(2, self.fdim),
            torch.nn.ReLU()
        )
        self.layers = torch.nn.ModuleList([
            MetaLayer(
                edge_model=EdgeModel(self.fdim),
                node_model=NodeModel(self.fdim),
                global_model=GlobalModel(self.fdim)
            ) for _ in range(cfg.n_blocks)
        ])
        self.decoder_e = torch.nn.Linear(self.fdim, 1)
        self.decoder_s = torch.nn.Linear(self.fdim, cfg.n_classes)

    def forward(self, data: BipartiteData) -> BipartiteData:
        x_s = self.encoder_s(data.x_s)
        x_t = self.encoder_t(data.x_t)
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        x_u = data.x_u

        for layer in self.layers:
            edge_index, x_s, x_t, x_u = layer(x_s, x_t, edge_attr, x_u, edge_index)
        return BipartiteData(edge_index, x_s, x_t, edge_attr, x_u)

    def predict_edge_time(self,
                           edge_attr: Tensor,
                           scale: float) -> Tensor:
        raw = self.decoder_e(edge_attr).squeeze(-1)
        return torch.relu(raw) * scale

    def predict_node_allocation(self,
                                 x_s: Tensor,
                                 scale: float) -> Tensor:
        probs = torch.softmax(self.decoder_s(x_s), dim=-1)
        return probs * scale

