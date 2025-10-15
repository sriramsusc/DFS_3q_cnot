
import torch, numpy as np
from train_autoencoder import AutoEncoder, sample_batch

device = "cpu"
model = AutoEncoder(latent_dim=64).to(device)
with torch.no_grad():
    batch = sample_batch(4).to(device)
    z, Uhat = model(batch)
    print("latent shape:", z.shape)
    # unitarity check
    Uh = Uhat.cpu().numpy()
    errs = []
    for k in range(Uh.shape[0]):
        e = np.linalg.norm(Uh[k].conj().T @ Uh[k] - np.eye(64))
        errs.append(float(e))
    print("avg unitarity error:", float(np.mean(errs)))
