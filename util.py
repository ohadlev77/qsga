from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING, Iterable

import numpy as np
import networkx as nx
from scipy.stats import ks_2samp, wasserstein_distance
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


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


def compute_eigenspectrum_ixn_laplacian(
    laplacian_hamiltonian: SparsePauliOp,
    sort: bool = False
) -> list[float]:
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
        sort: Whether to sort the eigenvalues (ascending order00).

    Returns:
        list[float]: The (sorted) eigenspectrum of the IX_n Laplacian Hamiltonian.
    """

    degree = np.real(laplacian_hamiltonian.coeffs[0]) # d

    num_nodes = laplacian_hamiltonian.dim[0] # |V|
    num_qubits = len(laplacian_hamiltonian.paulis[0].to_label()) # n
    
    eigenvalues = []
    for eigenvalue_index in range(num_nodes):

        eigenvalue = degree
        eigenvalue_index_binary = np.array(list(bin(eigenvalue_index)[2:].zfill(num_qubits)), dtype=int)

        for pauli in laplacian_hamiltonian.paulis[1:]:
            pauli_binary_mask = np.array(list(pauli.to_label().replace("I", "0").replace("X", "1")), dtype=int)

            exponent = np.dot(pauli_binary_mask, eigenvalue_index_binary)
            eigenvalue -= (-1) ** (exponent)

        eigenvalues.append(eigenvalue)

    return sorted(eigenvalues) if sort else eigenvalues


def compute_weighted_density(
    graph: nx.Graph | None = None,
    weighted_adjacency_matrix: NDArray[np.float64] | None = None
) -> float:
    """
    Compute the weighted density of a graph or adjacency matrix.

    The weighted density is defined as the sum of all edge weights divided by
    the number of possible undirected edges (|V| * (|V| - 1) / 2).

    Args:
        Exactly one of the following must be provided:
            graph: If provided, its weights will be used.
            weighted_adjacency_matrix: Numpy 2D array representing a weighted adjacency matrix.

    Returns:
        float: weighted density (total weight per possible edge).
    """
    
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
    required_unweighted_density: float,
    required_weighted_density: float | None = None,
    weights_bounds: tuple[float, float] | None = None,
    weights_distribution: NDArray[np.float64] = None,
    seed: int | None = None
) -> nx.Graph:
    """Generate a random weighted graph with specified density constraints.

    Args:
        num_nodes: Number of nodes in the graph.
        required_unweighted_density: Edge existence probability (0 to 1).
        Set one of the following:
            - required_weighted_density: Target weighted density. In this case, weights are given to edges
            such that the overall weighted density matches the target.
            - weights_bounds: Tuple (min, max) for uniform weight distribution. In this case, weights are sampled
            uniformly from the specified bounded interval.
            - weights_distribution: Specific distribution of weights to sample from. In this case, weights are drawn
            from the provided distribution.
        seed: Random seed for reproducibility.

    Returns:
        nx.Graph: Weighted graph with specified properties.

    Raises:
        ValueError: If both or neither of required_weighted_density and weights_bounds are set.
    """

    rng = np.random.default_rng(seed)
    
    graph: nx.Graph = nx.erdos_renyi_graph(n=num_nodes, p=required_unweighted_density, seed=seed)
    num_edges: int = graph.number_of_edges()
    possible_edges: int = (num_nodes * (num_nodes - 1)) / 2

    if (required_weighted_density is None) == (weights_bounds is None):
        raise ValueError("Set either required_weighted_density or weights_bounds, but not both.")
        
    if weights_distribution is None:
        if weights_bounds is None:
            expected_total_weight = required_weighted_density * possible_edges
            expected_weight_per_edge = expected_total_weight / num_edges

            mid_interval = expected_weight_per_edge
            half_interval = expected_weight_per_edge / 2
            weights_bounds = (mid_interval - half_interval, mid_interval + half_interval)

        weights = rng.uniform(*weights_bounds, size=num_edges)
    else:
        weights = rng.choice(weights_distribution, size=num_edges)

        # TODO REMOVE OR IMPLEMENT
        # len_weights = len(weights)
        # len_weights_distribution = len(weights_distribution)
        # if(len_weights > len_weights_distribution):
        #     res = compare_weight_distributions(weights[:len_weights_distribution], weights_distribution)
        # else:
        #     res = compare_weight_distributions(weights, weights_distribution[:len_weights])
        # print(res)

    for (u, v), weight in zip(graph.edges(), weights):
        graph[u][v]["weight"] = weight

    return graph


def compare_weight_distributions(
    w1: ArrayLike | Iterable[float],
    w2: ArrayLike | Iterable[float]
) -> dict[str, float]:
    """
    ### TODO CONSIDER TO REMOVE ###
    ### THIS FUNCTION HAS BEEN GENERATED BY CODEX ###

    Compare two equally sized weight samples with basic distributional distances.

    Args:
        w1: First set of weights (array-like); flattened internally via ``ravel()``.
        w2: Second set of weights (array-like); must have the same size as ``w1``.

    Returns:
        Dictionary with:
            - ks_stat: Kolmogorov-Smirnov statistic.
            - ks_pvalue: p-value for KS test of identical distributions.
            - wasserstein: 1D earth-mover distance between samples.
            - mean_diff: Difference in sample means (w1 - w2).
            - std_diff: Difference in sample standard deviations (w1 - w2).

    Notes:
        ``numpy.ravel`` flattens the input to 1D (view when possible, copy otherwise),
        so multi-dimensional arrays are treated as simple samples.
    """

    w1 = np.asarray(w1, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()

    if w1.size != w2.size:
        raise ValueError("Weight arrays must have the same size")

    ks_stat, ks_p = ks_2samp(w1, w2)
    wdist = wasserstein_distance(w1, w2)
    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "wasserstein": float(wdist),
        "mean_diff": float(w1.mean() - w2.mean()),
        "std_diff": float(w1.std(ddof=1) - w2.std(ddof=1)),
    }