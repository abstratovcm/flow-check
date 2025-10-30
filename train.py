import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True,
                    help="Path to JSON config file")
args = parser.parse_args()
cfg = json.load(open(args.config))

import os
import numpy as np
import torch
import time
import random
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.loss import MixturePathGeneralizedKL
from tqdm import tqdm
from data.chess_dataset import ChessBoardDataset, PIECE_ORDER
from model.vt_discrete_flow import VisionTransformerDiscreteFlow as DiscreteFlow
import losses
import csv

import chess
import chess.engine

def main():
    random.seed(cfg.get("seed", 0))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    piece_vals = {'.':0, 'P':1,'N':3,'B':3,'R':5,'Q':9,
                  'p':-1,'n':-3,'b':-3,'r':-5,'q':-9,
                  'K':0,'k':0}
    mat_sign = torch.tensor(
        [piece_vals[s] for s in PIECE_ORDER],
        device=device
    )
    print(f"Training on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if cfg.get("use_cp_loss", True):
        engine_path = cfg["engine_path"]
        stockfish = chess.engine.SimpleEngine.popen_uci(engine_path)
        stockfish.configure({"Threads": 4, "UCI_LimitStrength": False, "Hash": 2048})

    ds = ChessBoardDataset(
        cfg["parquet_path"],
        labels=cfg["labels"],
        label_column=cfg["label_column"]
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                   num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)


    warmup_epochs = cfg["warmup_epochs"]
    ramp_epochs = cfg["ramp_epochs"]
    d_model = cfg["d_model"]
    ff_dim = cfg["ff_dim"]
    if cfg.get("use_cfg", True) and cfg["num_labels"] > 1:
        model = DiscreteFlow(label_classes=cfg["num_labels"] + 1, d_model=d_model, ff_dim=ff_dim).to(device)
    else:
        model = DiscreteFlow(label_classes=cfg["num_labels"], d_model=d_model, ff_dim=ff_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(device=device.type)

    epochs = cfg["epochs"]
    total_steps = epochs * len(dl)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=100,
    )

    path_scheduler = PolynomialConvexScheduler(n=1.0)

    path = MixtureDiscreteProbPath(scheduler=path_scheduler)
    loss_fn = MixturePathGeneralizedKL(path=path)

    use_cfg   = cfg.get("use_cfg", True)
    eta = cfg["eta"]
    null_label = cfg["null_label"]

    labels = cfg["labels"]
    no_mat_ids    = [i for i,l in enumerate(labels) if "M^0" in l] # M^0 = Equal Material
    white_mat_ids = [i for i,l in enumerate(labels) if "M^+" in l] # M^+ = White Mat. Adv.
    black_mat_ids = [i for i,l in enumerate(labels) if "M^-" in l] # M^- = Black Mat. Adv.

    cp_bal_ids   = [i for i, l in enumerate(labels) if "E^0" in l] # E^0 = Equal Score
    cp_white_ids = [i for i, l in enumerate(labels) if "E^+" in l] # E^+ = White Score Adv.
    cp_black_ids = [i for i, l in enumerate(labels) if "E^-" in l] # E^- = Black Score Adv.

    no_mat_ids_t    = torch.tensor(no_mat_ids,    device=device)
    white_mat_ids_t = torch.tensor(white_mat_ids, device=device)
    black_mat_ids_t = torch.tensor(black_mat_ids, device=device)

    cp_bal_ids_t   = torch.tensor(cp_bal_ids,   device=device)
    cp_white_ids_t = torch.tensor(cp_white_ids, device=device)
    cp_black_ids_t = torch.tensor(cp_black_ids, device=device)

    lambda_min    = 0.0
    lambda_max    = 0.1

    log_interval = cfg.get("log_interval", 100)

    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    default_csv_path = os.path.join(ckpt_dir, "training_stats.csv")
    stats_csv = cfg.get("stats_csv_path", default_csv_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "mean_loss","std_loss","mean_king_loss","std_king_loss",
            "mean_piece_loss","std_piece_loss","mean_mat_loss","std_mat_loss",
            "mean_cp_loss","std_cp_loss","lr","epoch_time_s",
            "lambda_k","grad_norm","param_norm",
            "samples_per_sec","gpu_mem_gb"
        ])

    for epoch in range(epochs):
        batch_losses       = []
        batch_king_losses  = []
        batch_piece_losses = []
        batch_mat_losses   = []
        batch_cp_losses    = []
        grad_norms  = []
        param_norms = []
        if   epoch <= warmup_epochs - 1:
            lambda_k = 0.0
        elif epoch <= warmup_epochs + ramp_epochs - 1:
            frac       = (epoch - warmup_epochs + 1) / ramp_epochs
            lambda_k = lambda_min + frac * (lambda_max - lambda_min)
        else:
            lambda_k = lambda_max

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        total_batches = len(dl)
        epoch_loss = 0.0
        epoch_king_loss = 0.0
        epoch_piece_loss = 0.0
        epoch_mat_loss = 0.0
        epoch_cp_loss = 0.0

        pbar = tqdm(enumerate(dl, 1), total=total_batches, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x_1, label) in pbar:
            king_penalty = torch.tensor(0., device=device)
            piece_penalty = torch.tensor(0., device=device)
            mat_loss = torch.tensor(0., device=device)
            cp_pg_loss = torch.tensor(0., device=device)

            x_1 = x_1.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            B = x_1.size(0)
            x_0 = torch.randint(
                low=0,
                high=model.vocab_size,
                size=x_1.shape,
                device=device,
                dtype=torch.long
            )

            if use_cfg and cfg["num_labels"] > 1:
                mask = torch.rand(label.shape, device=device) < eta
                label_drop = label.clone()
                label_drop[mask] = null_label
            else:
                label_drop = label

            with autocast(device_type=device.type):
                t = torch.rand(x_1.size(0), device=device) * (1 - 1e-3)
                sample = path.sample(t=t, x_0=x_0, x_1=x_1)
                logits = model(sample.x_t, sample.t, label_drop)

                loss = loss_fn(
                    logits=logits,
                    x_1=sample.x_1,
                    x_t=sample.x_t,
                    t=sample.t
                )

            probs = torch.softmax(logits, dim=-1)

            w = 10*sample.t**9 - 9*sample.t**10

            if cfg.get("use_king_loss", True):
                king_penalty = losses.compute_king_loss(
                    probs, w
                )
                loss = loss + lambda_k * king_penalty
                epoch_king_loss += king_penalty.item()

            if cfg.get("use_piece_loss", True):
                piece_penalty = losses.compute_piece_loss(
                    probs, w
                )
                loss = loss + lambda_k * piece_penalty
                epoch_piece_loss += piece_penalty.item()

            if cfg.get("use_mat_loss", True):
                mat_loss = losses.compute_mat_loss(
                    probs, mat_sign, label, w, no_mat_ids_t,
                    white_mat_ids_t, black_mat_ids_t
                )
                loss = loss + lambda_k * mat_loss
                epoch_mat_loss += mat_loss.item()

            if cfg.get("use_cp_loss", True):
                cp_pg_loss = losses.compute_cp_loss(
                    logits, label, sample, B, device,
                    stockfish, engine_path,
                    cp_bal_ids_t, cp_white_ids_t, cp_black_ids_t
                )
                loss = loss + lambda_k * cp_pg_loss
                epoch_cp_loss += cp_pg_loss.item()

            batch_losses.append(loss.item())
            batch_king_losses.append(king_penalty.item())
            batch_piece_losses.append(piece_penalty.item())
            batch_mat_losses.append(mat_loss.item())
            batch_cp_losses.append(cp_pg_loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.detach().norm(2).item()**2
            grad_norm = total_norm**0.5
            grad_norms.append(grad_norm)
            scaler.step(optimizer)
            scaler.update()
            param_norm = 0.0
            for p in model.parameters():
                param_norm += p.detach().norm(2).item()**2
            param_norm = param_norm**0.5
            param_norms.append(param_norm)
            lr_scheduler.step()

            current_lr = lr_scheduler.get_last_lr()[0]

            epoch_loss += loss.item()
            avg_loss = epoch_loss / batch_idx

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                pbar.set_postfix(
                    Loss=f"{avg_loss:.4f}",
                    King=f"{king_penalty.item():.4f}",
                    CP=f"{cp_pg_loss.item():.4f}",
                    LR=f"{current_lr:.2e}"
                )

        mean_epoch_loss = epoch_loss / total_batches
        mean_king_loss = epoch_king_loss / total_batches
        mean_piece_loss = epoch_piece_loss / total_batches
        mean_mat_loss = epoch_mat_loss / total_batches
        mean_cp_loss = epoch_cp_loss / total_batches
        epoch_time = time.time() - start_time
        epoch_grad_norm = float(np.mean(grad_norms))
        epoch_param_norm= float(np.mean(param_norms))
        samples_per_sec = (len(dl) * cfg["batch_size"]) / epoch_time
        gpu_mem_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if torch.cuda.is_available() else 0.0

        epoch_lr = lr_scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Mean Loss: {mean_epoch_loss:.4f} | "
            f"King-Loss: {mean_king_loss:.4f} | "
            f"Piece-Loss: {mean_piece_loss:.4f} | "
            f"Mat-Loss: {mean_mat_loss:.4f} (@λ={lambda_k:.3f}) | "
            f"CP-Loss: {mean_cp_loss:.4f} (@λ={lambda_k:.3f}) | "
            f"LR: {epoch_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        stats = {
            "mean_loss": np.mean(batch_losses),
            "std_loss":  np.std(batch_losses),
            "mean_king_loss":  np.mean(batch_king_losses),
            "std_king_loss":   np.std(batch_king_losses),
            "mean_piece_loss": np.mean(batch_piece_losses),
            "std_piece_loss":  np.std(batch_piece_losses),
            "mean_mat_loss":   np.mean(batch_mat_losses),
            "std_mat_loss":    np.std(batch_mat_losses),
            "mean_cp_loss":    np.mean(batch_cp_losses),
            "std_cp_loss":     np.std(batch_cp_losses),
            "lr":              lr_scheduler.get_last_lr()[0],
            "epoch_time_s":    epoch_time,
            "lambda_k":        lambda_k,
            "grad_norm":       epoch_grad_norm,
            "param_norm":      epoch_param_norm,
            "samples_per_sec": samples_per_sec,
            "gpu_mem_gb":      gpu_mem_gb,
        }

        with open(stats_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1] + [stats[k] for k in stats])

        save_interval = cfg.get("save_interval", 1)
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, ckpt_path)
            print(f" → checkpoint saved to {ckpt_path}")

    if cfg.get("use_cp_loss", True):
        stockfish.quit()

if __name__ == "__main__":
    main()
