from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LaplacianDecomposition(NamedTuple):
    diagonal: NDArray[np.float64] # 1D array of length |V|
    D: NDArray[np.float64] # 2D array of dimenions |V| X |V|
    A: NDArray[np.float64] # 2D array of dimenions |V| X |V|

        
def decompose_laplacian_matrix(L: NDArray[np.float64]) -> LaplacianDecomposition:
    """Decompose a graph Laplacian matrix into its components.

    Args:
        L: The input Laplacian matrix to decompose.

    Returns:
        LaplacianDecomposition: A named tuple containing:
            - diagonal: Vector of diagonal elements.
            - D: Degree matrix (diagonal matrix).
            - A: Adjacency matrix.
    """

    diagonal = np.diag(L)
    D = np.diag(diagonal)
    A = D - L
    
    return LaplacianDecomposition(diagonal, D, A)


def transform_laplacian_to_graph(L: NDArray[np.float64]) -> nx.Graph:
    """Form a graph object given its Laplacian matrix."""

    diagonal, D, A = decompose_laplacian_matrix(L)
    return nx.from_numpy_array(np.real(A))


def compute_eigenspectrum_ixn_laplacian(laplacian_hamiltonian: SparsePauliOp) -> list[float]:
    r"""Analytically compute the (trivial) eigenspectrum of the IX_n Laplacian Hamiltonian.
    The analytical form of the eigenvalues is:
        $$ \lambda_k(L) = d - \sum_{P \in S} (-1)^{k \cdot m_P} $$
    Where:
        * $k$ is the index of the eigenvalue.
        * $S$ is the set of $IX$ operators that the Laplacian Hamiltonian is built from.
        * $P$ is a a single $IX$ operator in $S$.
        * $m_P$ is the *binary mask* of $P$. For example, $P = IXIX \implies m_P = 0101$.

    Args:
        laplacian_hamiltonian: The IX_n Laplacian Hamiltonian in a sparse Pauli representation.

    Returns:
        list[float]: The (sorted) eigenspectrum of the IX_n Laplacian Hamiltonian.
    """

    degree = np.real(laplacian_hamiltonian.coeffs[0])

    num_nodes = laplacian_hamiltonian.dim[0]
    num_qubits = len(laplacian_hamiltonian.paulis[0].to_label())
    
    eigenvalues = []
    for eigenvalue_index in range(num_nodes):

        eigenvalue = degree
        eigenvalue_index_binary = np.array(list(bin(eigenvalue_index)[2:].zfill(num_qubits)), dtype=int)

        for pauli in laplacian_hamiltonian.paulis[1:]:
            pauli_binary_mask = np.array(list(pauli.to_label().replace("I", "0").replace("X", "1")), dtype=int)

            exponent = np.dot(pauli_binary_mask, eigenvalue_index_binary)
            eigenvalue -= (-1) ** (exponent)

        eigenvalues.append(eigenvalue)
    
    return sorted(eigenvalues)