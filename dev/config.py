import os
import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    """
    Configuration for training and model parameters.
    """
    # Device configuration
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Paths
    data_file: str = os.getenv("DATA_FILE", "../params/increasing.txt")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "models")
    figures_dir: str = os.getenv("FIGURES_DIR", "figures")

    # Graph and dataset parameters
    n_fibers: int = 2000
    n_classes: int = 12
    n_fields: int = 10
    total_time: float = 42.0

    # Model hyperparameters
    fdim: int = 10        # hidden feature dimension
    n_blocks: int = 4     # number of MetaLayer blocks

    # Training hyperparameters
    lr: float = 5e-4
    epochs: int = 200_000
    p_class: float = 0.1  # class over-allocation penalty
    p_fiber: float = 0.1  # fiber over-allocation penalty
    w_utils: float = 2000.0
    w_var: float = 1.0
    sharps: tuple = (0.0, 100.0)
    min_sharp: float = 50.0
    seed: int = 42