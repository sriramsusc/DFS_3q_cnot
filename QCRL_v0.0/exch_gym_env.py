import gym
from gym import spaces
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


def _block(U: jnp.ndarray) -> jnp.ndarray:
    return U[LOGICAL][:, LOGICAL]


def _fidelity(A: jnp.ndarray, B: jnp.ndarray) -> float:
    inner = jnp.vdot(A, B)
    return jnp.abs(inner) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))

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
        3. For each step if the fidelity of the 64x64 is increased by .2, +8 reward.
        4. For each step if the fidelity of the 9x9 is increased, +.5 to the reward.
        5. For each step if the fidelity of the 9x9 is increased by .1, +2 reward.
        6. For each step if the fidelity of the 9x9 is increased by .2, +3 reward.
        7. If the matrix matched 27 elements of the target in the following rows and columns [9,10,12,17,18,20,33,34,36], then +30 reward.
        8. For each step if the fidelity with the target matrix is reduced compared to the previous step, then -1 reward.
        9. For each step if the fidelity with the target matrix is reduced by .1 compared to the previous step, then -2 reward.
        10. For each step if the fidelity with the target matrix is reduced by .2 compared to the previous step, then -3 reward.
        11. For each step after the 10th step, step penalty of -1.
        12. If the matrix matches the entire 64d rows and columns of  the target matrix in the following indices [9,10,12,17,18,20,33,34,36], which would be 9*64*2 elements, +80 reward and exit.
        13. If the fidelity is greater than .99, +100 reward and exit.
    """
    f64_now = _fidelity(U_now, TARGET_FULL)
    f9_now  = _fidelity(_block(U_now), TARGET_BLOCK)
    f64_prev = _fidelity(U_prev, TARGET_FULL)
    f9_prev  = _fidelity(_block(U_prev), TARGET_BLOCK)
    df64, df9 = f64_now - f64_prev, f9_now - f9_prev
    r = 0.0
    # improvements
    if df64 > 0: r += 1.0
    if df64 >= .1: r += 5.0
    if df64 >= .2: r += 8.0
    if df9  > 0: r += .5
    if df9  >= .1: r += 2.0
    if df9  >= .2: r += 3.0
    # penalties
    if df64 <= 0: r -= 1.0
    if df64 <= -.1: r -= 2.0
    if df64 <= -.2: r -= 3.0
    if step > 10: r -= .25 * step
    # exact matches
    rows, cols = jnp.meshgrid(LOGICAL, LOGICAL, indexing="ij")
    mask27 = (TARGET_FULL[rows, cols] != 0)
    if jnp.allclose(U_now[rows, cols][mask27], TARGET_FULL[rows, cols][mask27], atol=1e-6):
        r += 30.0
    # full match
    if jnp.allclose(U_now[LOGICAL], TARGET_FULL[LOGICAL], atol=1e-6):
        r += 80.0
        done = True
    elif f64_now >= .99:
        r += 100.0
        done = True
    else:
        done = (step >= max_depth)
    return r, done, f64_now, f9_now

# ----------------------------------------------------------------------
# 2. Gym-compatible environment for DI-engine PDQN
# ----------------------------------------------------------------------
@ENV_REGISTRY.register('ExchangeCNOTEnvDI')
class ExchangeCNOTEnvDI(gym.Env):
    """
    Gym environment with parameterized actions: (pair_index, p)
    Compatible with DI-engine PDQN.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 max_depth: int = 18,
                 obs_mode: str = 'block',
                 cfg: dict | None = None,         # <── new
                 **_unused):                      # <── swallows any extras
        if cfg is not None:                       # launched by DI-engine
            max_depth = cfg.get('max_depth', max_depth)
            obs_mode  = cfg.get('obs_mode',  obs_mode)
        super().__init__()
        self.max_depth = max_depth
        assert obs_mode in ('block','full','both')
        self.obs_mode = obs_mode
        # action: discrete pair 0..4 and continuous p in [-1,1]
        discrete = spaces.Discrete(len(NEIGHBORS))
        cont = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=jnp.float32)
        self.action_space = spaces.Tuple((discrete, cont))
        self.reward_space = spaces.Box(low=-jnp.inf, high=jnp.inf,
                               shape=(), dtype=jnp.float32)
        # observation: flattened parts + step
        dim = 0
        if obs_mode in ('full','both'): dim += 64*64*2
        if obs_mode in ('block','both'): dim += 9*9*2
        dim += 1
        self.observation_space = spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(dim,), dtype=jnp.float32
        )
        # state
        self.U = None
        self.step_count = None

    def reset(self):
        # initialize unitary
        self.U = jnp.eye(2**N_PHYS, dtype=jnp.complex64)
        # align global phase
        phase = jnp.vdot(self.U, TARGET_FULL) / jnp.abs(jnp.vdot(self.U, TARGET_FULL))
        self.U *= jnp.conj(phase)
        self.step_count = 0
        obs_jax = _vec_jax(jnp.array(self.U), self.step_count, self.obs_mode)
        self._episode_return = 0.0
        return jnp.array(obs_jax)

    def step(self, action):
        """
        Single‐env step, but under the hood we feed a length‐1 batch
        into our jitted+vmapped JAX kernels for maximum throughput.
        """
        # ── 1. normalise input ─────────────────────────────────────────────
        if isinstance(action, dict):
            if 'action_type' in action:
                pair_idx = int(action['action_type'])
            elif 'action' in action:
                pair_idx = int(action['action'])
            elif 'action_index' in action:
                pair_idx = int(action['action_index'])
            else:
                raise ValueError(f"Un-recognised discrete key in {action}")

            if 'action_args' in action:
                p = float(jnp.asarray(action['action_args'])[0])
            elif 'param' in action:
                p = float(action['param'])
            else:
                raise ValueError(f"Missing continuous key in {action}")

        elif isinstance(action, (tuple, list, jnp.ndarray)):
            pair_idx = int(action[0])
            p        = float(action[1])
        else:
            raise TypeError(f"Unsupported action type {type(action)}")

        # ── 2. prepare batch of size 1 ────────────────────────────────────
        U_prev = self.U.copy()
        U_batch     = jnp.expand_dims(jnp.array(self.U),    axis=0)  # shape (1,8,8)
        Uprev_batch = jnp.expand_dims(jnp.array(U_prev),      axis=0)

        # ── 3. apply gate via vmap+jit ────────────────────────────────────
        Unext_batch = _apply_batched(U_batch, int(pair_idx), p)            # (1,8,8)
        self.U      = jnp.array(Unext_batch[0])                        # back to (8,8)

        # ── 4. update step count & compute reward ────────────────────────
        self.step_count += 1
        r, done, f64, f9 = reward_fn(self.U, U_prev, self.step_count, self.max_depth)
        self._episode_return += r

        # ── 5. compute obs via vmap+jit ───────────────────────────────────
        obs_batch = _vec_batched(Unext_batch, self.step_count, self.obs_mode)  # (1,163)
        obs       = jnp.array(obs_batch[0])

        # ── 6. build info & return ────────────────────────────────────────
        info = dict(fid64=f64, fid9=f9)
        if done:
            info['eval_episode_return'] = self._episode_return

        return Timestep(obs, r, done, info)

    
    # ------------------------------------------------------------------
    # Gym/DI-engine compatibility helpers
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None, dynamic_seed: bool = False):
        """
        DI-engine calls   env.seed(seed, dynamic_seed)
        so we must accept two positional arguments.

        `dynamic_seed` is a bool telling the manager to bump the seed every
        episode; we don’t need it inside the environment, but we keep the
        parameter to satisfy the signature.
        """
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
            # If you use jax random elsewhere you can also do
            # jax.random.PRNGKey(seed) here and store it.
        return [seed]

    
    # ─────────────────────────────────────────────────────────────
    # ✨  DI-engine glue  ✨
    # ─────────────────────────────────────────────────────────────
    @classmethod
    def create_collector_env_cfg(cls, cfg: EasyDict):
        """
        Return a list with `cfg.collector_env_num` dictionaries.
        Each element is the kwargs that will be forwarded to
        `cls(**kwargs)` inside every collector worker.
        """
        base = dict(max_depth=cfg.max_episode_steps,  # map DI field → ctor kw
                    obs_mode=cfg.get('obs_mode', 'block'))
        return [deepcopy(base) for _ in range(cfg.collector_env_num)]

    @classmethod
    def create_evaluator_env_cfg(cls, cfg: EasyDict):
        base = dict(max_depth=cfg.max_episode_steps,
                    obs_mode=cfg.get('obs_mode', 'block'))
        return [deepcopy(base) for _ in range(cfg.evaluator_env_num)]

    def render(self, mode='human'):
        print(f"Step: {self.step_count}, Fid64: {_fidelity(self.U, TARGET_FULL):.4f}")

    def close(self):
        pass
