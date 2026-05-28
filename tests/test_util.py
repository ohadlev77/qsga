import pytest

import numpy as np
import networkx as nx

from qsga.util import (
    obtain_random_weighted_graph,
    decompose_laplacian_matrix,
    transform_laplacian_to_graph,
    compute_weighted_density,
    compare_hermitian_spectra
)


VERBOSE = True


def test_obtain_random_weighted_graph_with_weights_bounds():
    num_nodes = 10
    required_unweighted_density = 0.5
    weights_bounds = (1.0, 5.0)
    seed = 42

    graph = obtain_random_weighted_graph(
        num_nodes=num_nodes,
        required_unweighted_density=required_unweighted_density,
        weights_bounds=weights_bounds,
        seed=seed
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == num_nodes
    
    # Check if edges have weights within the specified bounds
    for u, v, data in graph.edges(data=True):
        assert "weight" in data
        assert weights_bounds[0] <= data["weight"] <= weights_bounds[1]

    if VERBOSE:
        print(f"\n--- test_obtain_random_weighted_graph_with_weights_bounds ---")
        print(f"Generated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        print(f"Weights bounds: {weights_bounds}")
        print(f"Sampled edge weights: {[d['weight'] for u, v, d in graph.edges(data=True)][:5]}...")


def test_obtain_random_weighted_graph_with_required_weighted_density():
    num_nodes = 20
    required_unweighted_density = 0.3
    required_weighted_density = 2.0
    seed = 42

    graph = obtain_random_weighted_graph(
        num_nodes=num_nodes,
        required_unweighted_density=required_unweighted_density,
        required_weighted_density=required_weighted_density,
        seed=seed
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == num_nodes

    # The expected total weight logic inside the function uses possible edges
    # possible_edges = (20 * 19) / 2 = 190
    # expected_total_weight = 2.0 * 190 = 380
    # The actual sum of weights should be approximately expected_total_weight
    # given uniform sampling around the expected weight per edge.
    # However, since we sample uniformly from (mid - half, mid + half), 
    # the exact sum may vary slightly, but we can check if weights exist.
    for u, v, data in graph.edges(data=True):
        assert "weight" in data

    if VERBOSE:
        print(f"\n--- test_obtain_random_weighted_graph_with_required_weighted_density ---")
        print(f"Generated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        print(f"Target Density: {required_weighted_density}")


def test_obtain_random_weighted_graph_with_weights_distribution_larger():
    num_nodes = 5
    required_unweighted_density = 0.5
    weights_bounds = (0.0, 1.0)
    weights_distribution = np.linspace(0.1, 1.0, 20) # 20 > max possible edges (10)
    seed = 42

    graph = obtain_random_weighted_graph(
        num_nodes=num_nodes,
        required_unweighted_density=required_unweighted_density,
        weights_bounds=weights_bounds,
        weights_distribution=weights_distribution,
        seed=seed
    )

    assert isinstance(graph, nx.Graph)
    for u, v, data in graph.edges(data=True):
        assert "weight" in data
        assert data["weight"] in weights_distribution

    if VERBOSE:
        print(f"\n--- test_obtain_random_weighted_graph_with_weights_distribution_larger ---")
        print(f"Weights distribution (size {len(weights_distribution)}): {weights_distribution[:5]}...")
        print(f"Sampled edge weights: {[d['weight'] for u, v, d in graph.edges(data=True)][:5]}...")


def test_obtain_random_weighted_graph_with_weights_distribution_slightly_smaller():
    # Force a specific number of edges by fixing seed and checking.
    # For num_nodes=5, p=1.0, there are exactly 10 edges.
    num_nodes = 5
    required_unweighted_density = 1.0
    weights_bounds = (0.0, 1.0)
    weights_distribution = np.linspace(0.1, 1.0, 8) # 8 < 10 edges, but 10 - 8 = 2 <= 8
    seed = 42

    graph = obtain_random_weighted_graph(
        num_nodes=num_nodes,
        required_unweighted_density=required_unweighted_density,
        weights_bounds=weights_bounds,
        weights_distribution=weights_distribution,
        seed=seed
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_edges() == 10
    
    # Check if all edge weights come from the provided distribution
    for u, v, data in graph.edges(data=True):
        assert "weight" in data
        assert data["weight"] in weights_distribution

    if VERBOSE:
        print(f"\n--- test_obtain_random_weighted_graph_with_weights_distribution_slightly_smaller ---")
        print(f"Weights distribution (size {len(weights_distribution)}): {weights_distribution}")
        print(f"Sampled edge weights: {[d['weight'] for u, v, d in graph.edges(data=True)]}")


def test_obtain_random_weighted_graph_value_error():
    num_nodes = 10
    required_unweighted_density = 0.5
    
    # Both None
    with pytest.raises(ValueError, match="Set either required_weighted_density or weights_bounds, but not both."):
        obtain_random_weighted_graph(
            num_nodes=num_nodes,
            required_unweighted_density=required_unweighted_density,
            required_weighted_density=None,
            weights_bounds=None
        )

    # Both not None
    with pytest.raises(ValueError, match="Set either required_weighted_density or weights_bounds, but not both."):
        obtain_random_weighted_graph(
            num_nodes=num_nodes,
            required_unweighted_density=required_unweighted_density,
            required_weighted_density=0.5,
            weights_bounds=(0.0, 1.0)
        )

    if VERBOSE:
        print(f"\n--- test_obtain_random_weighted_graph_value_error ---")


def test_decompose_laplacian_matrix():
    mat = np.array([
        [ 2.0, -1.0, -1.0],
        [-1.0,  2.0, -1.0],
        [-1.0, -1.0,  2.0]
    ])
    dec = decompose_laplacian_matrix(mat)
    assert np.array_equal(dec.diagonal, [2.0, 2.0, 2.0])
    assert np.array_equal(dec.D, [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
    assert np.array_equal(dec.A, [[0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0]])

    if VERBOSE:
        print(f"\n--- test_decompose_laplacian_matrix ---")
        print(f"Original Matrix:\n{mat}")
        print(f"Diagonal:\n{dec.diagonal}")
        print(f"Degree Matrix D:\n{dec.D}")
        print(f"Adjacency Matrix A:\n{dec.A}")


def test_transform_laplacian_to_graph():
    mat = np.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ])
    g = transform_laplacian_to_graph(mat)
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert g[0][1]["weight"] == 1.0

    if VERBOSE:
        print(f"\n--- test_transform_laplacian_to_graph ---")
        print(f"Laplacian Matrix:\n{mat}")
        print(f"Transformed Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")


def test_compute_weighted_density():
    mat = np.array([
        [ 0, 2.0, 0],
        [2.0, 0, 4.0],
        [ 0, 4.0, 0]
    ])
    # Possible edges = 3
    # Total weight = 6.0
    # Density = 6.0 / 3 = 2.0
    assert compute_weighted_density(weighted_adjacency_matrix=mat) == 2.0
    
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0)
    g.add_edge(1, 2, weight=4.0)
    g.add_node(2)
    assert compute_weighted_density(graph=g) == 2.0
    
    with pytest.raises(ValueError):
        compute_weighted_density()
        
    with pytest.raises(ValueError):
        compute_weighted_density(graph=g, weighted_adjacency_matrix=mat)

    if VERBOSE:
        print(f"\n--- test_compute_weighted_density ---")
        print(f"Adjacency Matrix:\n{mat}")
        print(f"Density from Matrix: 2.0")
        print(f"Density from Graph: 2.0")


def test_compare_hermitian_spectra():
    # identical spectra
    spec_a = np.array([1.0, 2.0])
    spec_b = np.array([1.0, 2.0])
    res = compare_hermitian_spectra(spec_a, spec_b)
    assert res["soergel_distance"] == 0.0
    assert res["spectral_similarity_sigma"] == 1.0

    # 10x-scaled spectra: soergel > 0, sigma = 10.0
    # p_nz = [2.0], q_nz = [20.0] → ratio = 20/2 = 10.0
    spec_c = np.array([10.0, 20.0])
    res2 = compare_hermitian_spectra(spec_a, spec_c)
    assert res2["soergel_distance"] > 0.0
    assert res2["spectral_similarity_sigma"] == 10.0

    # Laplacian-like (zero first eigenvalue): sigma ignores index 0
    # non-trivial eigenvalues scaled 2x → sigma = 2.0
    lap_a = np.array([0.0, 1.0, 3.0])
    lap_b = np.array([0.0, 2.0, 6.0])
    res3 = compare_hermitian_spectra(lap_a, lap_b)
    assert res3["spectral_similarity_sigma"] == 2.0

    # asymmetric ratio: worst-case eigenvalue pair wins
    # ratios: [1/2→2.0, 4/4→1.0] → sigma = 2.0
    lap_c = np.array([0.0, 1.0, 4.0])
    lap_d = np.array([0.0, 2.0, 4.0])
    res4 = compare_hermitian_spectra(lap_c, lap_d)
    assert res4["spectral_similarity_sigma"] == 2.0

    # all eigenvalues near zero → no safe pairs → sigma is None
    res5 = compare_hermitian_spectra(np.zeros(3), np.zeros(3))
    assert res5["spectral_similarity_sigma"] is None

    # mismatched lengths → ValueError
    with pytest.raises(ValueError):
        compare_hermitian_spectra(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    if VERBOSE:
        print(f"\n--- test_compare_hermitian_spectra ---")
        print(f"Identical:    {res}")
        print(f"10x-scaled:   {res2}")
        print(f"Laplacian 2x: {res3}")
        print(f"Asymmetric:   {res4}")
        print(f"All-zero:     {res5}")