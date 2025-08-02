import numpy as np
from typing import Dict, Sequence

MASK_LEN = 5  # number of neighbour pairs


def dict_to_vec(action: Dict[str, float | Sequence[float]] | Dict) -> np.ndarray:
    """Convert PDQN action dict to 6D vector."""
    pair_idx = int(
        action.get("action_type", action.get("action", action.get("action_index", 0)))
    )
    p_val = action.get("action_args", action.get("param", [0.0]))
    p = float(p_val[0] if isinstance(p_val, (list, tuple, np.ndarray)) else p_val)
    vec = np.zeros(MASK_LEN + 1, dtype=np.float32)
    vec[pair_idx] = 1.0
    vec[-1] = p
    return vec


def vec_to_dict(vec: Sequence[float]) -> Dict:
    """Convert 6D vector back to PDQN action dict."""
    if len(vec) != MASK_LEN + 1:
        raise ValueError(f"Expected vector of length {MASK_LEN + 1}")
    pair_idx = int(np.argmax(vec[:MASK_LEN]))
    p = float(vec[-1])
    return {"action_type": pair_idx, "action_args": [p]}