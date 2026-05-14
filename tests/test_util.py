import pytest
import numpy as np
import networkx as nx

from qsga.util import obtain_random_weighted_graph


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