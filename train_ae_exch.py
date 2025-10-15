# train_ae_exchange.py
# Train a convolutional autoencoder on (N,2,64,64) exchange unitaries stored in a single .npz.
# - Loss: MSE(Re/Im) + λ * ||U_rec^† U_rec - I||_F^2
# - Metrics: MSE, Frobenius reconstruction error, trace overlap |Tr(U†U_hat)|/d, unitarity error
# - Latent dimension is configurable; try 4, 8, 16, 32

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------- Dataset ---------------------------

class ExchangeNPZDataset(Dataset):
    def __init__(self, npz_path, mmap_mode="r"):
        # Memory-map so we never slurp the whole array into RAM
        self.data = np.load(npz_path, mmap_mode=mmap_mode)
        self.U = self.data["U"]        # shape (N,2,64,64), float32
        self.meta = self.data["meta"]  # shape (N,3): (i, j, p)
        assert self.U.ndim == 4 and self.U.shape[1:] == (2,64,64), "Expected (N,2,64,64)"
        # No normalization by default; these are exactly real/imag parts.

    def __len__(self):
        return self.U.shape[0]

    def __getitem__(self, idx):
        x = self.U[idx]             # (2,64,64), float32
        m = self.meta[idx]          # (3,)
        x = torch.from_numpy(x)     # torch.float32
        return x, torch.from_numpy(m.astype(np.float32))

# --------------------------- Model -----------------------------

class ConvAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # Encoder: (B,2,64,64) -> latent_dim
        # Keep strides modest to preserve detail; channels grow to 256.
        self.enc = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 2x2
            nn.ReLU(inplace=True),
        )
        self.enc_out_dim = 256 * 2 * 2
        self.to_latent = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: latent_dim -> (B,2,64,64)
        self.from_latent = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16,  kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,  2,   kernel_size=4, stride=2, padding=1), # 64x64
            # no activation: we want unconstrained real/imag
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        h = self.from_latent(z)
        h = h.view(z.size(0), 256, 2, 2)
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z

# ------------------------ Loss / Metrics -----------------------

def complex_from_2ch(x):
    # x: (B,2,64,64) -> complex (B,64,64)
    return torch.complex(x[:,0], x[:,1])

def fro_error(U, Uh):
    # mean Frobenius norm of difference
    return torch.linalg.matrix_norm(U - Uh, ord='fro', dim=(-2,-1)).mean()

def unitarity_error(Uh):
    # ||Uh^† Uh - I||_F averaged over batch
    B, d, _ = Uh.shape
    I = torch.eye(d, device=Uh.device, dtype=Uh.dtype).expand(B, d, d)
    return torch.linalg.matrix_norm(torch.matmul(Uh.conj().transpose(-1,-2), Uh) - I,
                                    ord='fro', dim=(-2,-1)).mean()

def trace_overlap(U, Uh):
    # |Tr(U^† Uh)| / d averaged
    d = U.shape[-1]
    prod = torch.matmul(U.conj().transpose(-1,-2), Uh)
    tr = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1)
    return (tr.abs() / d).mean()

# ------------------------- Train Loop --------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = ExchangeNPZDataset(args.data)
    N = len(ds)
    val_n = max(int(N * args.val_split), 1)
    train_n = N - val_n
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = ConvAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))
    mse = nn.MSELoss()

    best_val = float("inf")
    patience = args.patience
    patience_left = patience

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"ae_lat{args.latent_dim}.pt")

    for epoch in range(1, args.epochs+1):
        # ---- Train ----
        model.train()
        tr_loss = tr_mse = tr_unit = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                xhat, _ = model(xb)
                # reconstruction loss on 2 channels
                rec = mse(xhat, xb)
                # unitarity penalty on complex reconstruction
                Uh = complex_from_2ch(xhat)
                unit = unitarity_error(Uh)
                loss = rec + args.lambda_unit * unit
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * xb.size(0)
            tr_mse  += rec.item() * xb.size(0)
            tr_unit += unit.item() * xb.size(0)

        tr_loss /= train_n
        tr_mse  /= train_n
        tr_unit /= train_n

        # ---- Validate ----
        model.eval()
        va_mse = va_fro = va_unit = 0.0
        va_tr_overlap = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                xhat, _ = model(xb)
                rec = mse(xhat, xb)
                U  = complex_from_2ch(xb)
                Uh = complex_from_2ch(xhat)
                va_mse += rec.item() * xb.size(0)
                va_fro += fro_error(U, Uh).item() * xb.size(0)
                va_unit += unitarity_error(Uh).item() * xb.size(0)
                va_tr_overlap += trace_overlap(U, Uh).item() * xb.size(0)

        va_mse /= val_n
        va_fro /= val_n
        va_unit /= val_n
        va_tr_overlap /= val_n

        print(f"[{epoch:03d}] "
              f"train: loss={tr_loss:.3e} mse={tr_mse:.3e} unit={tr_unit:.3e} | "
              f"val: mse={va_mse:.3e} fro={va_fro:.3e} unit={va_unit:.3e} "
              f"overlap={va_tr_overlap:.6f}")

        # Early stopping on validation MSE
        if va_mse + args.lambda_unit * va_unit < best_val:
            best_val = va_mse + args.lambda_unit * va_unit
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_mse": va_mse,
                "val_unit": va_unit
            }, ckpt_path)
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    print(f"Best checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="exchange_6q_all.npz")
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--lambda_unit", type=float, default=1e-4)  # tiny unitarity nudge
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--patience", type=int, default=7)
    args = p.parse_args([])  # ← if running in a notebook; remove for CLI
    train(args)
