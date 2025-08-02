import jax, jax.numpy as jnp
import math

from ex_operations import exchange_gate_nqubits
from ex_operations import exchange_generators

from gymnax.environments import environment, spaces
from flax import struct
import jax, jax.numpy as jnp
from functools import reduce
from typing import Tuple
from fw_target import U_circuit                                

# ----------------------------------------------------------------------
# 1.  Static problem data
# ----------------------------------------------------------------------
N_PHYS = 6
NEIGHBORS = jnp.array([[0, 1],
                       [1, 2],
                       [2, 3],
                       [3, 4],
                       [4, 5]], dtype=jnp.int32)   # 5 nearest pairs

# logical sub-space indices (0-based) ─ rows/cols 
LOGICAL = jnp.array([9,10,12,17,18,20,33,34,36], dtype=jnp.int32)

# ----------  target unitary  -----------------------------------------
# IMPORT YOUR 64×64 TARGET HERE
from fw_target import U_circuit as TARGET_FULL           # (64,64) complex64
TARGET_BLOCK = TARGET_FULL[LOGICAL][:, LOGICAL]          # (9,9)
H_BASE = jnp.stack(
    [exchange_generators(N_PHYS, int(i), int(j)) for i, j in NEIGHBORS.tolist()],
    axis=0,                                             # shape (5, 64, 64)
)


# ----------------------------------------------------------------------
# 2.  Small helpers (JAX-friendly, no global state)
# ----------------------------------------------------------------------
def _block(U):                               # 9×9 logical block
    return U[LOGICAL][:, LOGICAL]


def _fidelity(A: jnp.ndarray, B: jnp.ndarray) -> jnp.float32:
    """|⟨A,B⟩| / (‖A‖‖B‖)  ϵ [0,1]"""
    inner = jnp.vdot(A, B)
    return jnp.abs(inner) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))


SEL = LOGICAL                                 # for the 27-element mask
ROWS, COLS = jnp.meshgrid(SEL, SEL, indexing="ij")
MASK27 = (TARGET_FULL[ROWS, COLS] != 0)        # (9,9) boolean


# ----------------------------------------------------------------------
# 3.  Environment data-classes
# ----------------------------------------------------------------------
@struct.dataclass(frozen=True)
class EnvState:
    U:        jnp.ndarray
    step:     jnp.int32
    fid64:    jnp.float32     # last-computed full-matrix fidelity
    fid9:     jnp.float32     # last-computed block fidelity



@struct.dataclass
class EnvParams:
    max_depth: int = 18
    dense_obs: bool = False
    obs_mode: str = "block"   # "block" | "full" | "both"

# ----------------------------------------------------------------------
# 4.  Reward function (implements the 13 rules)
# ----------------------------------------------------------------------
def reward_fn(U_now, U_prev, step):
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
    f9_now = _fidelity(_block(U_now), TARGET_BLOCK)
    f64_prev = _fidelity(U_prev, TARGET_FULL)
    f9_prev = _fidelity(_block(U_prev), TARGET_BLOCK)

    df64 = f64_now - f64_prev
    df9 = f9_now - f9_prev

    r = 0.0
    # 1-3
    r += jax.lax.select(df64 > 0, 1.0, 0.0)
    r += jax.lax.select(df64 >= 0.10, 5.0, 0.0)
    r += jax.lax.select(df64 >= 0.20, 8.0, 0.0)
    # 4-6
    r += jax.lax.select(df9 > 0, 0.5, 0.0)
    r += jax.lax.select(df9 >= 0.10, 2.0, 0.0)
    r += jax.lax.select(df9 >= 0.20, 3.0, 0.0)
    # 8-10 penalties
    r -= jax.lax.select(df64 < 0, 1.0, 0.0)
    r -= jax.lax.select(df64 <= -0.10, 2.0, 0.0)
    r -= jax.lax.select(df64 <= -0.20, 3.0, 0.0)
    # 11 step penalty
    r -= 0.25* step if step > 10 else 0.0

    # 7: 27-element exact match
    match27 = jnp.all(
        jnp.isclose(U_now[ROWS, COLS][MASK27],
                    TARGET_FULL[ROWS, COLS][MASK27],
                    atol=1e-6)
    )
    r += jax.lax.select(match27, 30.0, 0.0)

    # 12: all 9×64×2 real/imag entries of those rows identical
    full_match = jnp.all(jnp.isclose(U_now[SEL], TARGET_FULL[SEL], atol=1e-6))
    done_by_match = full_match
    r += jax.lax.select(done_by_match, 80.0, 0.0)

    # 13: fidelity ≥ .99
    done_by_fidelity = f64_now >= 0.99
    r += jax.lax.select(done_by_fidelity, 100.0, 0.0)

    done = jnp.logical_or(done_by_match, done_by_fidelity)
    return r.astype(jnp.float32), done, f64_now.astype(jnp.float32), f9_now.astype(jnp.float32)


# ----------------------------------------------------------------------
# 5.  The Gymnax environment
# ----------------------------------------------------------------------
class ExchangeCNOTEnv(environment.Environment):

    # ---- constructor -------------------------------------------------- #
    def __init__(self, params: EnvParams = EnvParams()):
        self._params = params
        super().__init__()
        
        # sub-spaces reused by observation_space / action_space
        self._pair_space  = spaces.Discrete(NEIGHBORS.shape[0])             # 5
        self._angle_space = spaces.Box(
            low  = -jnp.ones(()),        # lower bound  scalar
            high =  jnp.ones(()),        # upper bound  scalar
            shape= (),                   # *scalar* continuous parameter p
            dtype=jnp.float32,
        )


    # ---- spaces ------------------------------------------------------- #
    # read-only accessor that mirrors the Gymnax interface
    @property
    def default_params(self) -> EnvParams:
        return self._params

    @property
    def observation_space(self):
        mode = self.default_params.obs_mode
        if mode == "block":
            dim = 9 * 9 * 2 + 1
        elif mode == "full":
            dim = 64 * 64 * 2 + 1
        elif mode == "both":
            dim = 64 * 64 * 2 + 9 * 9 * 2 + 1
        else:
            raise ValueError(f"Unknown obs_mode {mode}")
        high = jnp.ones(dim, dtype=jnp.float32)
        return spaces.Box(-high, high)


    @property
    def action_space(self):
        """Tuple (pair_id, p)."""
        return spaces.Tuple((self._pair_space, self._angle_space))

    # ---- helpers ------------------------------------------------------ #
    @staticmethod
    @jax.jit
    def _apply(U, pair_id, p):
        H = H_BASE[pair_id]                                # gather (JIT-safe)
        U_gate = jax.scipy.linalg.expm(-1j * jnp.pi * p * H)
        return U_gate @ U
    def _vec(self, U, step):
        mode = self.default_params.obs_mode
        parts = []
        if mode in ("full", "both"):
            parts += [U.real.flatten(), U.imag.flatten()]
        if mode in ("block", "both"):
            blk = _block(U)
            parts += [blk.real.flatten(), blk.imag.flatten()]
        parts += [jnp.array([step], dtype=jnp.float32)]
        return jnp.concatenate(parts).astype(jnp.float32)

    # ---- reset -------------------------------------------------------- #
    def reset(self, key: jax.random.PRNGKey,
              params: EnvParams | None = None) -> Tuple[EnvState, jnp.ndarray]:
        p = self.default_params if params is None else params
        U0 = jnp.eye(2 ** N_PHYS, dtype=jnp.complex64)
        U0 = jnp.eye(2 ** N_PHYS, dtype=jnp.complex64)
        phase = jnp.vdot(U0, TARGET_FULL)/jnp.abs(jnp.vdot(U0, TARGET_FULL))
        fid64_0 = _fidelity(U0*phase.conj(), TARGET_FULL)
        fid9_0 = _fidelity(_block(U0), TARGET_BLOCK)
        obs0 = self._vec(U0, 0)
        return EnvState(U0, 0, fid64_0, fid9_0), obs0

    # ---- step --------------------------------------------------------- #
    def step(self, key: jax.random.PRNGKey, state: EnvState, action,
             params: EnvParams | None = None):
        """action = (pair_id, array([p]))"""
        p_cfg = self.default_params if params is None else params

        pair_id, p_val = action               # p_val is length-1 array
        p_scalar = jnp.squeeze(p_val)            

        U_next = self._apply(state.U, pair_id, p_scalar)
        step_no = state.step + 1

        # reward & termination
        r, done_intrinsic, f64, f9 = reward_fn(U_next, state.U, step_no)
        done = jnp.logical_or(done_intrinsic, step_no == p_cfg.max_depth)

        obs = self._vec(U_next, step_no)
        new_state = EnvState(U_next, step_no, f64, f9)
        return new_state, obs, r, done, {}