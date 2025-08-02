import jax, jax.numpy as jnp
from functools import reduce

from scipy.linalg import expm

I = jnp.eye(2, dtype=complex)
X = jnp.array([[0, 1], [1, 0]], dtype=complex)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
Z = jnp.array([[1, 0], [0, -1]], dtype=complex)

def exchange_generators(n,i,j):
    def _kron_all(mats):
        """Kronecker product of a list of matrices."""
        return reduce(jnp.kron, mats)
    
    def two_qubit_term(pauli):
        mats = [I] * n          # start with identities
        mats[i] = mats[j] = pauli
        return _kron_all(mats)

    XX = two_qubit_term(X)
    YY = two_qubit_term(Y)
    ZZ = two_qubit_term(Z)
    
    return (XX + YY + ZZ) / 4.0  # Heisenberg exchange Hamiltonian
     

def exchange_gate_nqubits(n: int, p: float, i: int, j: int):
    if not (0 <= i < n and 0 <= j < n and i != j):
        raise ValueError("Indices i and j must be distinct and in [0 .. n-1].")   

    def _kron_all(mats):
        """Kronecker product of a list of matrices."""
        return reduce(jnp.kron, mats)

    def two_qubit_term(pauli):
            mats = [I] * n          # start with identities
            mats[i] = mats[j] = pauli
            return _kron_all(mats)

    XX = two_qubit_term(X)
    YY = two_qubit_term(Y)
    ZZ = two_qubit_term(Z)
    H_swap_gen = (XX + YY + ZZ) / 4.0         # Heisenberg exchange Hamiltonian
    U = expm(-1j * jnp.pi * p * H_swap_gen)   # e^(-i π p H_swap)

    return U