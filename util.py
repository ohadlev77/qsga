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

        
def decompose_laplacian_matrix(L: NDArray[np.float64] | SparsePauliOp) -> LaplacianDecomposition:
    """Decompose a graph Laplacian matrix into its components.

    Args:
        L: The input Laplacian matrix to decompose.

    Returns:
        LaplacianDecomposition: A named tuple containing:
            - diagonal: Vector of diagonal elements.
            - D: Degree matrix (diagonal matrix).
            - A: Adjacency matrix.
    """

    if isinstance(L, SparsePauliOp):
        L = L.to_matrix().real

    diagonal = np.diag(L)
    D = np.diag(diagonal)
    A = D - L
    
    return LaplacianDecomposition(diagonal, D, A)


def transform_laplacian_to_graph(L: NDArray[np.float64] | SparsePauliOp) -> nx.Graph:
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


def compute_weighted_density(
    graph: nx.Graph | None = None,
    weighted_adjacency_matrix: NDArray[np.float64] | None = None
) -> float:
    
    forbidden_cases = [
        graph is None and weighted_adjacency_matrix is None,
        graph is not None and weighted_adjacency_matrix is not None
    ]
    if any(forbidden_cases):
        raise ValueError("Set either graph or weighted_adjacency_matrix")
        
    if weighted_adjacency_matrix is None:
        A = nx.to_numpy_array(graph)
    else:
        A = weighted_adjacency_matrix
    
    weights_sum = np.sum(A) / 2
    num_nodes = A.shape[0]
    possible_edges = (num_nodes * (num_nodes - 1)) / 2
    
    weighted_density = weights_sum / possible_edges
    
    return weighted_density


def obtain_random_weighted_graph(
    num_nodes: int,
    required_unweighted_density: float, # Which is going to be the edge-existence probability
    required_weighted_density: float | None = None,
    weights_bounds: tuple[float, float] | None = None,
    seed: int | None = None
) -> nx.Graph:
    """TODO COMPLETE."""

    rng = np.random.default_rng(seed)
    
    graph = nx.erdos_renyi_graph(n=num_nodes, p=required_unweighted_density, seed=seed)
    num_edges = graph.number_of_edges()
    possible_edges = (num_nodes * (num_nodes - 1)) / 2

    if (required_weighted_density is None) == (weights_bounds is None):
        raise ValueError("Set either required_weighted_density or weights_bounds, but not both.")
    
    if weights_bounds is None:
        expected_total_weight = required_weighted_density * possible_edges
        expected_weight_per_edge = expected_total_weight / num_edges

        mid_interval = expected_weight_per_edge
        half_interval = expected_weight_per_edge / 2
        weights_bounds = (mid_interval - half_interval, mid_interval + half_interval)

    for (u, v) in graph.edges():
        graph[u][v]["weight"] = rng.uniform(*weights_bounds)

    return graph