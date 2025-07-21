import os
import matplotlib.pyplot as plt
import numpy as np
from config import Config

def plot_history(history: dict, cfg: Config) -> None:
    """
    Plot training loss and utility history.
    """
    epochs = np.arange(1, len(history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.figures_dir, 'loss_history.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history['utility'])
    plt.xlabel('Epoch')
    plt.ylabel('Min Class Completion')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.figures_dir, 'utility_history.png'))
    plt.close()

def plot_final_distributions(fiber_times: np.ndarray, cfg: Config) -> None:
    """
    Plot histogram of final fiber times.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(fiber_times, bins=30, alpha=0.7)
    plt.axvline(x=cfg.total_time, color='red', linestyle='--', label='Total Time')
    plt.xlabel('Fiber Time')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.figures_dir, 'fiber_time_histogram.png'))
    plt.close()