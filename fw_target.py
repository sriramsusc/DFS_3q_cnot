import math
import jax.numpy as jnp
import numpy as np
from ex_operations import exchange_gate_nqubits

p1 = math.acos(-1/math.sqrt(3)) / math.pi
p2 = math.asin(1/3) / math.pi
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

U_circuit = jnp.eye(64, dtype=complex)
for (p, (i, j)) in gate_specs:
    U_gate = exchange_gate_nqubits(6, p, i, j)
    U_circuit = U_gate @ U_circuit