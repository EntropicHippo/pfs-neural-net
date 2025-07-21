import os
import argparse
import random
import torch
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime

from config import Config
from gnn import GNN
from data_loader import load_bipartite_data
from loss_utils import compute_loss
from round_utils import smart_round
from plot_utils import plot_history, plot_final_distributions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str, model: GNN, optimizer: Adam):
    """
    Load model & optimizer state if checkpoint exists.
    """
    start_epoch = 1
    best_utility = -float('inf')
    history = {'loss': [], 'utility': []}
    best_ckpt = None
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=model.encoder_s[0].weight.device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt['epoch'] + 1
        best_utility = ckpt['best_utility']
        history = ckpt['history']
        best_ckpt = ckpt
    return start_epoch, best_utility, history, best_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN for ML4PS allocation task"
    )
    parser.add_argument(
        "--checkpoint",
        default="gnn_checkpoint.pth",
        help="Checkpoint filename"
    )
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    data = load_bipartite_data(cfg).to(cfg.device)
    model = GNN(cfg).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    checkpoint_path = os.path.join(cfg.checkpoint_dir, args.checkpoint)
    start_epoch, best_utility, history, best_ckpt = load_checkpoint(
        checkpoint_path, model, optimizer
    )

    # Training loop
    for epoch in tqdm(
        range(start_epoch, cfg.epochs + 1),
        desc="Training"
    ):
        model.train()
        optimizer.zero_grad()

        out_data = model(data)
        sharp = cfg.sharps[0] + (
            cfg.sharps[1] - cfg.sharps[0]
        ) * (epoch - 1) / cfg.epochs
        loss, utility = compute_loss(model, out_data, cfg, sharp)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['utility'].append(
            utility.item() if isinstance(utility, torch.Tensor) else utility
        )

        # Checkpoint on improvement
        if utility > best_utility and sharp > cfg.min_sharp:
            best_utility = utility
            best_ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_utility': best_utility,
                'history': history
            }
            torch.save(best_ckpt, checkpoint_path)

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(cfg.checkpoint_dir, 'gnn_final.pth')
    )

    # Plot training history
    plot_history(history, cfg)

    # Final evaluation and rounding
    best_model = GNN(cfg).to(cfg.device)
    if best_ckpt is not None:
        best_model.load_state_dict(best_ckpt['model_state'])
    out_data = best_model(data)
    (loss_f, util_f,
     comp, n_prime,
     fiber_time, time_spent,
     var_term) = compute_loss(
        best_model, out_data, cfg, sharpness=cfg.sharps[1],
        final_output=True
    )

    class_info_np = data.x_t.detach().cpu().numpy()
    time_matrix = time_spent.reshape(cfg.n_fibers, cfg.n_classes)
    rounded_data, updated_util, _, updated_fiber_times, updated_completion = smart_round(
        time_matrix, comp, class_info_np, cfg
    )

    # Write final results
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(cfg.figures_dir, f'final_results_{now}.txt')
    with open(results_path, 'w') as f:
        f.write(f"TIMESTAMP: {now}\n")
        f.write(f"Best Utility: {updated_util:.4f}\n")
        f.write(f"Final Completion: {updated_completion}\n")

    # Plot final fiber-time distribution
    plot_final_distributions(updated_fiber_times, cfg)

if __name__ == "__main__":
    main()
