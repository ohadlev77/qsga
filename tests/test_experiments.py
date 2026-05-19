import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

from qsga.experiments import (
    SingleExperimentConfiguration,
    ExperimentConfigurations,
    GraphMetadata,
    GraphData,
    LaplacianHamiltoniansWorkshop
)


VERBOSE = True


def test_single_experiment_configuration():
    config = SingleExperimentConfiguration(
        n_num_qubits=2,
        d_skeleton_regularity=3,
        max_skeleton_locality=4,
        num_perturbations=5,
        max_perturbation_locality=6,
        seed=42
    )
    assert config.num_nodes == 4
    assert str(config) == "q2-d3-sl4-np5-m6-s42"

    config_no_seed = SingleExperimentConfiguration(
        n_num_qubits=2,
        d_skeleton_regularity=3,
        max_skeleton_locality=4,
        num_perturbations=5,
        max_perturbation_locality=6
    )
    assert str(config_no_seed) == "q2-d3-sl4-np5-m6"

    if VERBOSE:
        print(f"\n--- test_single_experiment_configuration ---")
        print(f"Config details:\n{config}")
        print(f"Config without seed details:\n{config_no_seed}")


def test_experiment_configurations():
    configs = ExperimentConfigurations(
        n_num_qubits=[2, 3],
        d_skeleton_regularity=[3],
        max_skeleton_locality=[4],
        num_perturbations=[lambda x: x * 2],
        max_perturbation_locality=[6],
        seed=[42]
    )
    configs_list = list(configs)
    assert len(configs_list) == 2
    
    assert configs_list[0].n_num_qubits == 2
    assert configs_list[0].num_perturbations == 4
    
    assert configs_list[1].n_num_qubits == 3
    assert configs_list[1].num_perturbations == 6

    if VERBOSE:
        print(f"\n--- test_experiment_configurations ---")
        print(f"Total combinations: {len(configs_list)}")
        print(f"First config:\n{configs_list[0]}")
        print(f"Second config:\n{configs_list[1]}")


def test_graph_metadata():
    metadata = GraphMetadata(
        num_nodes=4,
        num_edges=5,
        unweighted_density=0.5,
        weighted_density=0.8
    )
    assert metadata.num_nodes == 4
    assert metadata.num_edges == 5
    assert metadata.unweighted_density == 0.5
    assert metadata.weighted_density == 0.8

    if VERBOSE:
        print(f"\n--- test_graph_metadata ---")
        print(f"Metadata Object: {metadata}")


def test_graph_data_from_sparse_op():
    op = SparsePauliOp(["I", "X"], [1, -1])
    data = GraphData(laplacian_sparse_obj=op, num_laplacian_paulis=2)
    
    assert data.metadata.num_nodes == 2
    assert data.num_laplacian_paulis == 2
    assert data.laplacian_dense_matrix.shape == (2, 2)
    assert np.allclose(data.laplacian_dense_matrix, [[1, -1], [-1, 1]])
    assert isinstance(data.graph_obj, nx.Graph)
    assert str(data) == "GraphData(nodes=2, edges=1, density=1.0000, weighted_density=1.0000, paulis=2, commuting_groups=1)"

    if VERBOSE:
        print(f"\n--- test_graph_data_from_sparse_op ---")
        print(f"Input Operator:\n{op}")
        print(f"Resulting Dense Matrix:\n{data.laplacian_dense_matrix}")


def test_graph_data_from_graph():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    data = GraphData(graph_obj=graph, num_laplacian_paulis=-1)
    
    assert data.metadata.num_nodes == 2
    assert data.num_laplacian_paulis == 2
    assert np.allclose(data.laplacian_dense_matrix, [[1, -1], [-1, 1]])
    assert str(data) == "GraphData(nodes=2, edges=1, density=1.0000, weighted_density=1.0000, paulis=2, commuting_groups=1)"

    if VERBOSE:
        print(f"\n--- test_graph_data_from_graph ---")
        print(f"Input Graph Edges: {graph.edges(data=True)}")
        print(f"Resulting Dense Matrix:\n{data.laplacian_dense_matrix}")

    
def test_laplacian_hamiltonians_workshop_init():
    ec = ExperimentConfigurations(
        n_num_qubits=[2],
        d_skeleton_regularity=[1],
        max_skeleton_locality=[1],
        num_perturbations=[1, lambda x: x],
        max_perturbation_locality=[1]
    )
    workshop = LaplacianHamiltoniansWorkshop(configurations=ec)
    assert workshop.metadata["total_configurations"] == 2
    
    # Check that callables are converted to strings in metadata
    num_perturbations_meta = workshop.metadata["configurations"]["num_perturbations"]
    assert isinstance(num_perturbations_meta[1], str)
    assert "lambda" in num_perturbations_meta[1] or "function" in num_perturbations_meta[1]

    if VERBOSE:
        print(f"\n--- test_laplacian_hamiltonians_workshop_init ---")
        print(f"Workshop metadata: {workshop.metadata}")


def test_plot_soergel_distance(tmp_path):
    from pathlib import Path
    
    # Define a tiny experiment configuration with valid parameter values
    ec = ExperimentConfigurations(
        n_num_qubits=[3],
        d_skeleton_regularity=[3],
        max_skeleton_locality=[2],
        num_perturbations=[1, 2],
        max_perturbation_locality=[2, 3],
        seed=[42]
    )
    
    workshop = LaplacianHamiltoniansWorkshop(configurations=ec)
    
    # Run the experiment and analysis
    workshop.perform_experiment()
    workshop.analyze_results()
    
    # Set the run directory in metadata to use tmp_path
    workshop.metadata["run_metadata"] = {"run_dir": str(tmp_path)}
    
    # Plot Soergel distance and verify it saves the plot
    workshop.plot_soergel_distance(show_only=False)
    
    plot_file = tmp_path / "soergel_distance_vs_locality.png"
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0
    
    if VERBOSE:
        print("\n--- test_plot_soergel_distance ---")
        print(f"Successfully generated Soergel distance plot at: {plot_file}")
