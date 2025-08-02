import torch
import numpy as np
from types import SimpleNamespace
from inverse_rl.airl import AIRL
from inverse_rl.discriminator_network import G, H


class AIRLReward:
    """Wrapper around the AIRL discriminator to provide rewards."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        model_path: str | None = None,
    ):
        args = SimpleNamespace(
            state_only=False,
            layer_num=2,
            hidden_dim=128,
            activation_function=torch.relu,
            last_activation=None,
            gamma=0.99,
            lr=1e-3,
        )
        self.model = AIRL(None, device, state_dim, action_dim, args)
        self.model.to(device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def __call__(
        self,
        log_prob: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> float:
        with torch.no_grad():
            r = self.model.get_reward(
                torch.as_tensor(log_prob, dtype=torch.float32, device=self.device),
                torch.as_tensor(state, dtype=torch.float32, device=self.device),
                torch.as_tensor(action, dtype=torch.float32, device=self.device),
                torch.as_tensor(next_state, dtype=torch.float32, device=self.device),
                torch.as_tensor(done, dtype=torch.float32, device=self.device),
            )
            return float(r.cpu().numpy())