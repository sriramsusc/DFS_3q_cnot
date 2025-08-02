# test_exchange_cnot_env.py
import math
import numpy as np
import pytest
import math, logging, numpy as np, pytest
from exch_gym_env import ExchangeCNOTEnvDI, NEIGHBORS   # adjust import path if needed

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cnot‐env")

p1 = math.acos(-1 / math.sqrt(3)) / math.pi      # ≈ 0.304086723
p2 = math.asin( 1 / 3)            / math.pi      # ≈ 0.108253176

gate_specs = [
    ( 1+p1,  [3,4] ),
    # ( p1,    [3,4] ),
    ( p2,    [4,5] ),
    ( 0.5,   [2,3] ),
    ( 1.0,   [3,4] ),
    (-0.5,   [2,3] ),
    (-0.5,   [4,5] ),
    ( 1.0,   [1,2] ),
    (-0.5,   [3,4] ),
    (-0.5,   [2,3] ),
    ( 1.0,   [4,5] ),
    (-0.5,   [1,2] ),
    ( 0.5,   [3,4] ),
    (-0.5,   [2,3] ),
    ( 1.0,   [4,5] ),
    ( 1.0,   [1,2] ),
    (-0.5,   [3,4] ),
    (-0.5,   [2,3] ),
    (-0.5,   [4,5] ),
    ( 1.0,   [3,4] ),
    ( 0.5,   [2,3] ),
    ( 1-p2,  [4,5] ),
    # ( -p1,   [3,4] ),
    ( 1-p1,  [3,4] ),
]


def pair_to_index(pair):
    for idx, (i, j) in enumerate(NEIGHBORS):
        if pair in ([i, j], [j, i]): return idx
    raise ValueError

env = ExchangeCNOTEnvDI(max_depth=30, obs_mode="block")
obs = env.reset()
cum_r = 0.0
print("step | pair | p        | r   | fid64   | fid9")
for k, (p, pair) in enumerate(gate_specs, 1):
    ts = env.step({"action_type": pair_to_index(pair), "action_args": [p]})
    cum_r += ts.reward
    print(f"{k:4d} | {pair} | {p:+.6f} | {ts.reward:+.3f} | "
          f"{ts.info['fid64']:.6f} | {ts.info['fid9']:.6f}")
    if ts.done:
        break

print("-"*64)
print(f"terminated: {ts.done}   total reward: {cum_r:+.3f}")
print(f"final fidelities  F64={ts.info['fid64']:.6f}  F9={ts.info['fid9']:.6f}")
env.close()


