import gym
from gym import spaces
import math
import numpy as np
import jax, jax.numpy as jnp
from jax import jit, vmap, lax
from scipy.linalg import expm
from ding.utils import ENV_REGISTRY
from ex_operations import exchange_generators
from fw_target import U_circuit as TARGET_FULL
from copy import deepcopy
from easydict import EasyDict
from functools import partial
from collections import namedtuple

# ----------------------------------------------------------------------
# 1. Static problem data
# ----------------------------------------------------------------------
N_PHYS = 6
NEIGHBORS = jnp.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [4, 5]], dtype=jnp.int32)
LOGICAL = jnp.array([9,10,12,17,18,20,33,34,36], dtype=jnp.int32)
TARGET_BLOCK = TARGET_FULL[LOGICAL][:, LOGICAL]
# precompute generators
H_BASE = jnp.stack(
    [jnp.array(exchange_generators(N_PHYS, int(i), int(j))) for i,j in NEIGHBORS],
    axis=0
)
# A minimal container with the fields DI-engine expects
Timestep = namedtuple('Timestep', ['obs', 'reward', 'done', 'info'])

INVALID_PENALTY = -20.0

def _to_np(x, dtype=np.float32):          #  helper
    return np.asarray(x, dtype=dtype)

def _block(U: jnp.ndarray) -> jnp.ndarray:
    return U[LOGICAL][:, LOGICAL]

def _block_9x64(M):
    """rows = LOGICAL, all columns"""
    return M[LOGICAL, :]

def _block_64x9(M):
    """all rows, cols = LOGICAL"""
    return M[:, LOGICAL]


def _fidelity(A: jnp.ndarray, B: jnp.ndarray) -> float:

    """
    Computes the fidelity between two square matrices A and B.
    Fidelity is defined as:
    F(A, B) = (Tr(A†A) + |Tr(B†A)|²) / (n * (n + 1))

    updated from using just falttened inner product divided by product of norms.

    # inner = jnp.vdot(A, B)
    # return jnp.abs(inner) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))

    """

    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")

    n = A.shape[0]

    # --- core computation ----------------------------------------------------
    term1 = jnp.trace(A.conj().T @ A)                 # Tr(A†A)
    term2 = jnp.trace(B.conj().T @ A)                 # Tr(B†A)
    fidelity_val = (term1 + abs(term2)**2) / (n * (n + 1))

    # If you prefer a plain Python float:
    return float(np.real_if_close(fidelity_val))
# ----------------------------------------------------------------------
def frobenius_diff(A: np.ndarray, B: np.ndarray) -> float:
    diff = A - B
    return np.linalg.norm(A-B, ord='fro')
# ----------------------------------------------------------------------


# jit-compiled apply
@jit
def _apply_jax(U: jnp.ndarray, pair: int, p: float) -> jnp.ndarray:
    H = H_BASE[pair]
    U_gate = jax.scipy.linalg.expm(-1j * jnp.pi * p * H)
    return U_gate @ U

@partial(jit, static_argnums=(2,))
def _vec_jax(U: jnp.ndarray, step: int, mode: str) -> jnp.ndarray:
    parts = []
    if mode in ("full", "both"):
        parts += [jnp.real(U).ravel(), jnp.imag(U).ravel()]
    if mode in ("block", "both"):
        blk = U[LOGICAL][:, LOGICAL]
        parts += [jnp.real(blk).ravel(), jnp.imag(blk).ravel()]
    parts += [jnp.array([step], dtype=jnp.float32)]
    return jnp.concatenate(parts).astype(jnp.float32)
_apply_batched = jit(vmap(_apply_jax, in_axes=(0, None, None), out_axes=0))
_vec_batched   = jit(vmap(_vec_jax,   in_axes=(0, None, None), out_axes=0), static_argnums=(2,))


def reward_fn(U_now, U_prev, step, max_depth):
    """
        Implements the 10 bullet-points exactly (vectorised & JIT-safe).
        Returns  (reward, done)
        1. For each step if the fidelity of the 64x64 is increased , +1 to the reward.
        2. For each step if the fidelity of the 64x64 is increased by .1, +5 reward.
        3. For each step if the fidelity of the 64x64 is increased by .2, +7 reward.

        9x9 indices are [9,10,12,17,18,20,33,34,36](zero based).
        4. For each step if the fidelity of the 9x9 is increased, +.5 to the reward.
        5. For each step if the fidelity of the 9x9 is increased by .1, +2 reward.
        6. For each step if the fidelity of the 9x9 is increased by .2, +3 reward.

        9x64 block is [(9,x),(10,x),(12,x),(17,x),(18,x),(20,x),(33,x),(34,x),(36,x)] for x in range(64).
        7. For each step if the forbenius distance between the 9x64 block and the target matrix's 9x64 block is decresed compared to the previous step, then +2 reward.
        8. For each step if the forbenius distance between the 9x64 block and the target matrix's 9x64 block is decresed by >.2 compared to the previous step, then +4 reward.
        9. For each step if the forbenius distance between the 9x64 block and the target matrix's 9x64 block is decresed by 0 compared to the previous step, then +5 reward.
        64x9 block is [(x,9),(x,10),(x,12),(x,17),(x,18),(x,20),(x,33),(x,34),(x,36)] for x in range(64).
        10. For each step if the forbenius distance between the 64x9 block and the target matrix's 64x9 block is decresed compared to the previous step, then +2 reward.
        11. For each step if the forbenius distance between the 64x9 block and the target matrix's 64x9 block is decresed by >.2 compared to the previous step, then +4 reward.
        12. For each step if the forbenius distance between the 64x9 block and the target matrix's 64x9 block is decresed by 0 compared to the previous step, then +5 reward.

        13. For each step if the forbenius distance between both the 9x64 block and 64x9 block and the target matrix's 9x64 block and the 64x9 block is 0, then +100 reward and exit.
        
        14. For each step if the forbenius distance between the 9x64 block and the target matrix's 9x64 block is increased compared to the previous step, then -1 reward.
        15. For each step if the forbenius distance between the 64x9 block and the target matrix's 9x64 block is increased compared to the previous step, then -1 reward.
        16. For each step if the fidelity with the target matrix is decresed compared to the previous step, then -1 reward.
        17. For each step if the fidelity with the target matrix is decresed by .1 compared to the previous step, then -2 reward.
        18. For each step if the fidelity with the target matrix is decresed by .2 compared to the previous step, then -3 reward.
        19. For each step after the 8th step, step penalty of -(sqrt(step_count)).
        20. If the fidelity is greater than .96 or forb norm rectangular blocks < .03, +100 reward and exit.
    """
    # ---------- fidelity 64 × 64 & 9 × 9 ----------------------------------
    f64_now  = _fidelity(U_now,  TARGET_FULL)
    f64_prev = _fidelity(U_prev, TARGET_FULL)

    f9_now   = _fidelity(_block(U_now),  TARGET_BLOCK)
    f9_prev  = _fidelity(_block(U_prev), TARGET_BLOCK)

    df64 = f64_now - f64_prev
    df9  = f9_now  - f9_prev

    # ---------- Frobenius distances for 9×64 and 64×9 ---------------------
    frob9x64_now  = float(jnp.linalg.norm(_block_9x64(U_now)  - _block_9x64(TARGET_FULL)))
    frob9x64_prev = float(jnp.linalg.norm(_block_9x64(U_prev) - _block_9x64(TARGET_FULL)))
    d_frob9x64    = frob9x64_prev - frob9x64_now          # >0 means “closer”

    frob64x9_now  = float(jnp.linalg.norm(_block_64x9(U_now)  - _block_64x9(TARGET_FULL)))
    frob64x9_prev = float(jnp.linalg.norm(_block_64x9(U_prev) - _block_64x9(TARGET_FULL)))
    d_frob64x9    = frob64x9_prev - frob64x9_now

    # ---------- build reward ---------------------------------------------
    r = 0.0
    # 1-3  (64×64 fidelity improvements)
    if df64  > 0:    r += 1.0
    if df64 >= .1:   r += 2.0
    if df64 >= .2:   r += 3.0
    # 4-6  (9×9 block fidelity improvements)
    if df9   > 0:    r += 0.5
    if df9  >= .1:   r += 1.0
    if df9  >= .2:   r += 2.0
    # 7-9  (9×64 Frobenius ↓)
    if d_frob9x64  > 0  :   r += 1.0
    if d_frob9x64  > .09:   r += 2.0
    if d_frob9x64  > .2 :   r += 4.0
    if frob9x64_now == 0:   r += 8.0
    # 10-12 (64×9 Frobenius ↓)
    if d_frob64x9  > 0  :   r += 1.0
    if d_frob64x9  > .09:   r += 2.0
    if d_frob64x9  > .2 :   r += 4.0
    if frob64x9_now == 0:   r += 8.0
    # 13 perfect match of both cross-blocks
    # if (frob9x64_now == 0) and (frob64x9_now == 0):
    if (frob9x64_now == 0) and (frob64x9_now == 0):
        r   += 100.0
        done = True
        return r, done, f64_now, f9_now, frob9x64_now, frob64x9_now
    # 14-15 penalties (Frobenius ↑)
    if d_frob9x64  < 0: r -= 1.0
    if d_frob64x9  < 0: r -= 1.0
    # 16-18 penalties (64×64 fidelity ↓)
    if df64  < 0:      r -= 1.0
    if df64 <= -.1:    r -= 2.0
    if df64 <= -.2:    r -= 3.0
    # 19 step-penalty after 8th step
    if step > 8:
        r -= math.sqrt(step)
    # 20 near-perfect overall fidelity
    # if f64_now >= .96 or (d_frob64x9 <= 0.03 and d_frob9x64 <= 0.03):
    if f64_now >= .96:
        r += 100.0
        done = True
    else:
        done = (step >= max_depth)

    # r+=1

    return r, done, f64_now, f9_now

# ----------------------------------------------------------------------
# 2. Gym-compatible environment for DI-engine PDQN
# ----------------------------------------------------------------------
MASK_LEN = len(NEIGHBORS)   
@ENV_REGISTRY.register('ExchangeCNOTEnvDI')
class ExchangeCNOTEnvDI(gym.Env):
    """
    Parameter-ised exchange-gate environment **forbidding consecutive
    repeats of the same qubit pair** via an action-mask.

    – The mask lives in both the observation tail **and** in `info["action_mask"]`.
    – If the agent violates the mask we return a penalty (or raise).
    – Added: neat episode-summary prints that show the gate sequence,
             cumulative reward and fidelity after every episode.
    """

    metadata = {"render.modes": ["human"]}

    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------
    def __init__(self, max_depth: int = 18, obs_mode: str = "block",
                 cfg: dict | None = None, **_unused):
        # DI-engine passes the whole cfg dict – pick out what we need
        if cfg is not None:
            max_depth = cfg.get("max_depth", max_depth)
            obs_mode  = cfg.get("obs_mode",  obs_mode)

        super().__init__()
        assert obs_mode in ("block", "full", "both")

        # ---------- episode-print helpers ----------
        self._ep_idx     = 0
        self._seq        = []
        # self._cum_reward = 0.0
        self._last_fid   = float("nan")   # filled at episode end
        # -------------------------------------------

        self.max_depth  = int(max_depth)
        self.obs_mode   = obs_mode

        # action = (discrete pair, continuous p ∈ [-2,2])
        self.action_space = spaces.Tuple((
            spaces.Discrete(MASK_LEN),
            spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=jnp.float32)
        ))
        self.reward_space = spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(), dtype=jnp.float32)

        # observation dim
        dim = 1                                      # step counter
        if obs_mode in ("full", "both"):   dim += 64 * 64 * 2
        if obs_mode in ("block", "both"):  dim += 9 * 9 * 2
        dim += MASK_LEN                               # 5-bit mask
        self.observation_space = spaces.Box(low=-jnp.inf, high=jnp.inf,
                                            shape=(dim,), dtype=jnp.float32)

        # env state
        self.U              = None
        self.step_count     = None
        self.last_pair      = None
        self.valid_mask     = None
        self._episode_return = 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_obs(self) -> jnp.ndarray:
        base = _vec_jax(jnp.array(self.U), self.step_count, self.obs_mode)
        return _to_np(jnp.concatenate([base, self.valid_mask]))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *_, **__) -> jnp.ndarray:
        """Start a fresh episode – and *after* the previous one, print summary."""
        # ------- print the previous episode (if any) -------
        if self._seq:           # only if we actually finished one episode
            print(f"Episode {self._ep_idx:03d} │ "
                  f"reward = {self._episode_return:+.3f} │ "
                  f"fidelity = {self._last_fid:.6f}\n"
                  f"Sequence: {self._seq}\n")
        # ------- clear trackers for the new episode -------
        self._ep_idx     += 1
        self._seq         = []
        # self._cum_reward  = 0.0
        self._last_fid    = float("nan")
        # ---------------------------------------------------

        self.U = jnp.eye(2 ** N_PHYS, dtype=jnp.complex64)
        phase  = jnp.vdot(self.U, TARGET_FULL)
        self.U *= jnp.conj(phase / jnp.abs(phase))

        self.step_count      = 0
        self.last_pair       = None
        self.valid_mask      = jnp.ones(MASK_LEN, dtype=jnp.float32)
        self._episode_return = 0.0
        return self._make_obs()

    def step(self, action):
        # --------------------------------------------------------------
        # 1. canonicalise incoming action
        # --------------------------------------------------------------
        if isinstance(action, dict):                # DI-engine style
            pair_idx = int(action.get("action_type",
                          action.get("action", action.get("action_index"))))
            p_val    = action.get("action_args", action.get("param", [0.0]))
            p        = float(p_val[0] if isinstance(p_val, (list, tuple, jnp.ndarray)) else p_val)
        else:                                       # tuple / list / ndarray
            pair_idx, p = int(action[0]), float(action[1])

        # --------------------------------------------------------------
        # 2. legal-action check (mask violation ⇒ penalty timestep)
        # --------------------------------------------------------------
        if self.valid_mask[pair_idx] == 0:
            self._episode_return += INVALID_PENALTY
            obs  = self._make_obs()               # mask unchanged
            info = dict(
                invalid_action=True,
                action_mask=_to_np(self.valid_mask),
                eval_episode_return=float(self._episode_return)
            )
            return Timestep(obs, float(INVALID_PENALTY), False, info)

        # --------------------------------------------------------------
        # 3. normal transition
        # --------------------------------------------------------------
        U_prev      = self.U
        U_batch     = self.U[jnp.newaxis, ...]             # (1,64,64)
        self.U      = _apply_batched(U_batch, pair_idx, p)[0]

        self.step_count += 1
        self.last_pair   = pair_idx
        self.valid_mask  = jnp.ones(MASK_LEN, dtype=jnp.float32).at[pair_idx].set(0)

        r, done, fid64, fid9 = reward_fn(self.U, U_prev, self.step_count, self.max_depth)
        self._episode_return += r

        # track episode details for printing
        self._seq.append((pair_idx, p))
        if done:
            self._last_fid = float(fid9)          # choose whichever fidelity you care about

        obs  = self._make_obs()
        info = dict(
            fid64=float(fid64),
            fid9=float(fid9),
            action_mask=_to_np(self.valid_mask),
            eval_episode_return=float(self._episode_return)
        )
        print(f"[debug] step {self.step_count}  raw_reward={r}")
        return Timestep(obs, float(r), done, info)

    # ------------------------------------------------------------------
    # misc glue
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None, dynamic_seed: bool = False):
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        return [seed]

    @classmethod
    def create_collector_env_cfg(cls, cfg: EasyDict):
        base = dict(max_depth=cfg.max_episode_steps,
                    obs_mode=cfg.get("obs_mode", "block"))
        return [deepcopy(base) for _ in range(cfg.collector_env_num)]

    @classmethod
    def create_evaluator_env_cfg(cls, cfg: EasyDict):
        base = dict(max_depth=cfg.max_episode_steps,
                    obs_mode=cfg.get("obs_mode", "block"))
        return [deepcopy(base) for _ in range(cfg.evaluator_env_num)]

    def render(self, mode="human"):
        print(f"Step {self.step_count:2d} | fidelity64 = {_fidelity(self.U, TARGET_FULL):.4f}")

    def close(self):
        pass
