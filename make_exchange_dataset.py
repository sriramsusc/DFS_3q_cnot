# make_exchange_dataset_single_npz.py
# Produces ONE concatenated .npz for 6-qubit exchange ops on adjacent pairs.

import os
import numpy as np
from tqdm import tqdm
from ex_operations import exchange_gate_nqubits

# ------------ Config ------------
n_qubits = 6
pairs = [(0,1),(1,2),(2,3),(3,4),(4,5)]
num_p = 5001                                 # >= 5000, evenly spaced, includes endpoints
p_values = np.linspace(-1.0, 1.0, num=num_p, endpoint=True, dtype=np.float64)

out_npz = "exchange_6q_all.npz"              # final single output file
tmp_U = "_tmp_U_memmap.npy"                  # temp memmap files on disk
tmp_meta = "_tmp_meta_memmap.npy"

dtype_store = np.float32
unitarity_check_every = 200                  # set None to skip quick unitarity checks
unitarity_tol = 1e-4
# ---------------------------------

d = 2 ** n_qubits  # 64
N = len(pairs) * len(p_values)

def to_2ch(U: np.ndarray) -> np.ndarray:
    """(64,64) complex -> (2,64,64) float32 [Re, Im]."""
    out = np.empty((2, d, d), dtype=dtype_store)
    out[0] = U.real.astype(dtype_store, copy=False)
    out[1] = U.imag.astype(dtype_store, copy=False)
    return out

def is_unitary(U: np.ndarray, tol: float) -> bool:
    I = np.eye(U.shape[0], dtype=U.dtype)
    err = np.linalg.norm(U.conj().T @ U - I, ord='fro')
    return err <= tol

# Prepare on-disk memmaps to stream writes without holding everything in RAM
U_mm = np.lib.format.open_memmap(tmp_U, mode="w+", dtype=dtype_store, shape=(N, 2, d, d))
meta_mm = np.lib.format.open_memmap(tmp_meta, mode="w+", dtype=np.float32, shape=(N, 3))

idx = 0
print(f"Generating {N} unitaries into memmaps...")
with tqdm(total=N, unit="U") as pbar:
    for (i, j) in pairs:
        for p in p_values:
            U = np.asarray(exchange_gate_nqubits(n_qubits, float(p), i, j))  # complex
            if (unitarity_check_every is not None) and (idx % unitarity_check_every == 0):
                if not is_unitary(U, unitarity_tol):
                    raise RuntimeError(f"Non-unitary at pair=({i},{j}), p={p:.6f}")

            U_mm[idx] = to_2ch(U)
            meta_mm[idx] = (i, j, p)
            idx += 1
            pbar.update(1)

# Flush memmaps to disk before packaging
del U_mm
del meta_mm

print(f"Writing single compressed archive: {out_npz} (this may take a bit)...")
# Reopen as memmap (read) so np.savez_compressed streams from disk
U_mm = np.lib.format.open_memmap(tmp_U, mode="r")
meta_mm = np.lib.format.open_memmap(tmp_meta, mode="r")

np.savez_compressed(out_npz, U=U_mm, meta=meta_mm,
                    desc=np.array(["U: (N,2,64,64) float32 [Re,Im]",
                                   "meta: (N,3) float32 rows [i,j,p]"],
                                  dtype=object))

# Clean up temp files
try:
    os.remove(tmp_U)
    os.remove(tmp_meta)
except OSError:
    pass

print("Done.")
print(f"Wrote: {out_npz}")
