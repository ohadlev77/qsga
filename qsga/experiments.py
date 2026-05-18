from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product, cycle
from pathlib import Path
from typing import Iterator, Iterable, TYPE_CHECKING, Callable

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from qsga import (
    GRAPH_TYPES,
    OBSOLETE_GRAPHS,
    GRAPHS_TO_PLOT_MAP,
    GRAPHS_COMPARISON_PAIR
)
from qsga.data_verifiers import is_valid_laplacian
from qsga.data_handling import save_dataset, load_dataset, _slugify
from qsga.util import (
    obtain_random_weighted_graph,
    compute_weighted_density,
    transform_laplacian_to_graph,
    compare_hermitian_spectra
)
from qsga.hamiltonian_generators import (
    obtain_skeleton_laplacian,
    obtain_random_perturbed_laplacian,
    PerturbationScalingMethod
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SingleExperimentConfiguration:
    """Configuration for a single quantum graph Laplacian generation experiment.
    
    Attributes:
        n_num_qubits: Number of qubits (determines number of nodes as $|V| = 2^n$).
        d_skeleton_regularity: Regularity of the skeleton graph.
        max_skeleton_locality: Maximum locality for skeleton Hamiltonian.
        num_perturbations: Number of perturbations to apply.
        max_perturbation_locality: Maximum locality for each perturbation Laplacian Hamiltonian.
        perturbation_weights_bounds: Optional bounds for perturbation weights.
        seed: Optional random seed for reproducibility.
    """

    n_num_qubits: int
    d_skeleton_regularity: int
    max_skeleton_locality: int
    num_perturbations: int
    max_perturbation_locality: int

    perturbation_weights_bounds: tuple[float, float] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.num_nodes = 2 ** self.n_num_qubits

    def __str__(self) -> str:
        """
        Generate a string representation of the configuration as a slugified identifier.
        Constructs a concise slug containing key configuration parameters:
            - n: number of qubits
            - d: skeleton regularity
            - sl: maximum skeleton locality
            - p: number of perturbations
            - pl: maximum perturbation locality
            - s: random seed (optional, included only if set)

        Returns:
            str: A slugified string identifier representing the configuration parameters.
        """
        
        base = (
            f"q{self.n_num_qubits}-"
            f"d{self.d_skeleton_regularity}-"
            f"sl{self.max_skeleton_locality}-"
            f"np{self.num_perturbations}-"
            f"m{self.max_perturbation_locality}"
        )

        if self.seed is not None:
            base += f"-s{self.seed}"
            
        return _slugify(base)


@dataclass(frozen=True)
class ExperimentConfigurations:
    """Configuration container for generating multiple experiment configurations.
    
    Holds lists of parameters that will be combined using `itertools.product`
    to create all possible `SingleExperimentConfiguration` instances.

    Attributes:
        n_num_qubits: List of qubit counts to test.
        d_skeleton_regularity: List of skeleton regularities to test.
        max_skeleton_locality: List of maximum skeleton localities to test.
        num_perturbations: List of perturbation counts to test.
        max_perturbation_locality: List of maximum perturbation localities to test.
        perturbation_weights_bounds: Optional list of weight bounds for perturbations.
        seed: Optional list of random seeds for reproducibility.
    """

    n_num_qubits: list[int]
    d_skeleton_regularity: list[int]
    max_skeleton_locality: list[int]
    num_perturbations: list[int | Callable[[int], int]]
    max_perturbation_locality: list[int]

    perturbation_weights_bounds: list[tuple[float, float] | None] | None = None
    seed: list[int] | None = None

    def __iter__(self) -> Iterator[SingleExperimentConfiguration]:
        """Generate all possible SingleExperimentConfiguration instances from the parameter lists."""

        for vals in product(
            self.n_num_qubits,
            self.d_skeleton_regularity,
            self.max_skeleton_locality,
            self.num_perturbations,
            self.max_perturbation_locality,
            self.perturbation_weights_bounds or [None],
            self.seed or [None],
        ):
            (
                n_num_qubits,
                d_skeleton_regularity,
                max_skeleton_locality,
                num_perturbations_val,
                max_perturbation_locality,
                perturbation_weights_bounds,
                seed
             ) = vals
            
            # The number of perturbations can be a function of the number of qubits
            if callable(num_perturbations_val):
                num_perturbations_val = num_perturbations_val(n_num_qubits)

            yield SingleExperimentConfiguration(
                n_num_qubits,
                d_skeleton_regularity,
                max_skeleton_locality,
                num_perturbations_val,
                max_perturbation_locality,
                perturbation_weights_bounds,
                seed
            )


@dataclass
class GraphMetadata:
    num_nodes: int
    num_edges: int
    unweighted_density: float
    weighted_density: float


@dataclass
class GraphData:
    """
    A dataclass that encapsulates graph data and its Laplacian matrix representation.
    This class handles the initialization and storage of graph-related objects,
    including sparse Pauli operator representations of the Laplacian matrix,
    the graph object itself, and computed metadata.

    Attributes:
        laplacian_sparse_obj (SparsePauliOp | None): A sparse Pauli operator representation
            of the Laplacian matrix. If provided, it will be validated and converted to
            multiple list representations. Defaults to None.
        graph_obj (nx.Graph | None): A NetworkX graph object. If not provided, it will be
            constructed from the Laplacian sparse object. Defaults to None.
        laplacian_pauli_repr (list[tuple[str, complex]]): A list representation of the
            Laplacian sparse object in Pauli basis. Generated during initialization if
            laplacian_sparse_obj is provided.
        laplacian_sparse_pauli_repr (list[tuple[str, list[int], complex]]): A sparse list
            representation of the Laplacian sparse object. Generated during initialization
            if laplacian_sparse_obj is provided.
        laplacian_dense_matrix (NDArray[np.float64]): The dense Laplacian matrix computed
            from the graph object as a NumPy array.
        metadata (GraphMetadata[int | float]): Computed metadata about the graph including
            the number of nodes, edges, and density metrics.

    Raises:
        ValueError: If the provided laplacian_sparse_obj is not a valid Laplacian matrix.

    Note:
        Either laplacian_sparse_obj or graph_obj must be provided; if only one is provided,
        the other will be derived from it.
    """

    laplacian_sparse_obj: SparsePauliOp | None = None
    num_laplacian_paulis: int | None = None
    num_commuting_groups: int | None = None
    graph_obj: nx.Graph | None = None
    laplacian_spectrum: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.laplacian_sparse_obj is not None:
            is_valid_laplacian(self.laplacian_sparse_obj, throw_exception=True)
            self.laplacian_pauli_repr: list[tuple[str, complex]] = self.laplacian_sparse_obj.to_list()
            self.laplacian_sparse_pauli_repr: list[tuple[str, list[int], complex]] = self.laplacian_sparse_obj.to_sparse_list()

        if self.graph_obj is None:
            self.graph_obj: nx.Graph = transform_laplacian_to_graph(self.laplacian_sparse_obj)
        self.laplacian_dense_matrix: NDArray[np.float64] = nx.laplacian_matrix(self.graph_obj).todense()

        # self.num_laplacian_paulis == -1 => A graph is given, not a Laplacian operator
        op = self.laplacian_sparse_obj
        if self.num_laplacian_paulis == -1 and self.laplacian_sparse_obj is None:
            op = SparsePauliOp.from_operator(self.laplacian_dense_matrix).simplify()
            self.num_laplacian_paulis = len(op)

        if op is not None:
            if self.num_laplacian_paulis < 10_000:
                self.num_commuting_groups = len(op.group_commuting())
            else:
                self.num_commuting_groups = "N/A"
        
        self.metadata: GraphMetadata[int | float] = GraphMetadata(
            num_nodes=self.graph_obj.number_of_nodes(),
            num_edges=self.graph_obj.number_of_edges(),
            unweighted_density=nx.density(self.graph_obj),
            weighted_density=compute_weighted_density(self.graph_obj)
        )


class LaplacianHamiltoniansWorkshop:
    """Generate and analyze Laplacian Hamiltonians with perturbations.
    
    This class orchestrates the full workflow of generating quantum graph Laplacians,
    applying perturbations, comparing with random graphs, analyzing spectral properties,
    and visualizing results.
    """

    @staticmethod
    def from_data(data_dir_path: Path | str) -> LaplacianHamiltoniansWorkshop:
        """Load experiment data and configurations from a previously saved run.
        
        Args:
            data_dir_path: Path to the directory containing saved experiment data.
            
        Returns:
            LaplacianHamiltoniansWorkshop: Restored experiment object with loaded data.
        """
        
        data, manifest_data, metadata = load_dataset(data_dir_path)

        configurations = ExperimentConfigurations(**metadata["configurations"])

        obj = LaplacianHamiltoniansWorkshop.__new__(LaplacianHamiltoniansWorkshop)
        
        for config_result in data:
            if "configuration" in config_result and isinstance(config_result["configuration"], dict):
                # num_nodes was calculated in post_init, so we remove it before recreating if it was saved
                cfg_dict = config_result["configuration"].copy()
                cfg_dict.pop("num_nodes", None)
                config_result["configuration"] = SingleExperimentConfiguration(**cfg_dict)

            for gtype in GRAPH_TYPES:
                if gtype in config_result and isinstance(config_result[gtype], dict):
                    b_dict = config_result[gtype]
                    gd = GraphData(
                        graph_obj=b_dict.get("graph_obj"),
                        laplacian_spectrum=b_dict.get("laplacian_spectrum")
                    )
                    
                    if "laplacian_obj" in b_dict:
                        gd.laplacian_dense_matrix = b_dict["laplacian_obj"]

                    if "laplacian_pauli_repr" in b_dict:
                        gd.num_laplacian_paulis = len(b_dict["laplacian_pauli_repr"])
                    else:
                        gd.num_laplacian_paulis = "N/A"
                        
                    gd.num_commuting_groups = "N/A"
                    config_result[gtype] = gd

        obj.data = data
        obj.configurations = configurations
        obj.manifest_data = manifest_data
        obj.metadata = metadata

        return obj

    def __init__(self, configurations: ExperimentConfigurations) -> None:
        """Initialize the experiment with configurations.
        
        Args:
            configurations: `ExperimentConfigurations` object containing all parameter combinations.
        """

        self.configurations = configurations
        self.data: list[dict[str, GraphData]] = []
        self.metadata = {
            "configurations": asdict(configurations),
            "total_configurations": len(list(configurations)),
            "graph_types": GRAPH_TYPES
        }

        # Converting functions to strings for proper JSON serialization
        for index, element in enumerate(self.metadata["configurations"]["num_perturbations"]):
            if callable(element):
                self.metadata["configurations"]["num_perturbations"][index] = str(element)

    def perform_experiment(self) -> None:
        """Generate all Laplacian graphs and compute their properties.
        
        For each configuration, generates:
        - Skeleton Laplacian graph.
        - Definite-order perturbed Laplacian (`PerturbationScalingMethod.LEFT`).
        - Random-order perturbed (ROP) Laplacian (`PerturbationScalingMethod.RANDOM_LEFT_RIGHT`).
        - Randomly Perturbed Scrambled Laplacian (`PerturbationScalingMethod.SCRAMBLE`).
        - Random graphs with matching densities and weights.
        """

        for config_index, config in enumerate(self.configurations):

            # Skeleton Laplacian
            skeleton_laplacian = obtain_skeleton_laplacian(
                n=config.n_num_qubits,
                d=config.d_skeleton_regularity,
                max_locality=config.max_skeleton_locality,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            skeleton_graph_data = GraphData(
                laplacian_sparse_obj=skeleton_laplacian,
                num_laplacian_paulis=len(skeleton_laplacian)
            )

            # Perturbed Laplacians
            kwargs = dict(
                skeleton_hamiltonian=skeleton_laplacian,
                num_perturbations=config.num_perturbations,
                max_perturbation_locality=config.max_perturbation_locality,
                random_perturbation_weights_bounds=config.perturbation_weights_bounds,
            )
            
            definite_order_perturbed_laplacian = obtain_random_perturbed_laplacian(
                **kwargs,
                perturbations_scaling_method=PerturbationScalingMethod.LEFT,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            definite_order_perturbed_graph_data = GraphData(
                laplacian_sparse_obj=definite_order_perturbed_laplacian,
                num_laplacian_paulis=len(definite_order_perturbed_laplacian)
            )
            
            random_order_perturbed_laplacian = obtain_random_perturbed_laplacian(
                **kwargs,
                perturbations_scaling_method=PerturbationScalingMethod.RANDOM_LEFT_RIGHT,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            random_order_perturbed_graph_data = GraphData(
                laplacian_sparse_obj=random_order_perturbed_laplacian,
                num_laplacian_paulis=len(random_order_perturbed_laplacian)
            )

            random_order_scrambled_perturbed_laplacian = obtain_random_perturbed_laplacian(
                **kwargs,
                perturbations_scaling_method=PerturbationScalingMethod.SCRAMBLE,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            random_order_scrambled_perturbed_graph_data = GraphData(
                laplacian_sparse_obj=random_order_scrambled_perturbed_laplacian,
                num_laplacian_paulis=len(random_order_scrambled_perturbed_laplacian)
            )

            config_data: dict[str, int | str | GraphData] = {
                "config_index": config_index,
                "configuration": config,
                "skeleton_graph": skeleton_graph_data,
                "definite_order_perturbed_graph": definite_order_perturbed_graph_data,
                "random_order_perturbed_graph": random_order_perturbed_graph_data,
                "random_order_scrambled_perturbed_graph": random_order_scrambled_perturbed_graph_data,
            }

            # Same density Erdos-Renyi graph as the scrambled perturbed graph
            scrambled_like_random_graph = obtain_random_weighted_graph(
                num_nodes=random_order_scrambled_perturbed_graph_data.metadata.num_nodes,
                required_unweighted_density=random_order_scrambled_perturbed_graph_data.metadata.unweighted_density,
                required_weighted_density=random_order_scrambled_perturbed_graph_data.metadata.weighted_density,
                seed=config.seed
            )
            scrambled_like_random_graph_data = GraphData(
                graph_obj=scrambled_like_random_graph,
                num_laplacian_paulis=-1
            ) 
            config_data["scrambled_like_random_graph"] = scrambled_like_random_graph_data

            # Same density Erdos-Renyi graph as the scrambled perturbed graph + SAME WEIGHTS DISTRIBUTION
            weights = np.abs(
                np.triu(random_order_scrambled_perturbed_graph_data.laplacian_dense_matrix, k=1).flatten()
            )
            weights = weights[weights != 0]
            scrambled_like_random_graph_same_weights = obtain_random_weighted_graph(
                num_nodes=random_order_scrambled_perturbed_graph_data.metadata.num_nodes,
                required_unweighted_density=random_order_scrambled_perturbed_graph_data.metadata.unweighted_density,
                required_weighted_density=random_order_scrambled_perturbed_graph_data.metadata.weighted_density,
                seed=config.seed,
                weights_distribution=weights
            )
            scrambled_like_random_graph_same_weights_data = GraphData(
                graph_obj=scrambled_like_random_graph_same_weights,
                num_laplacian_paulis=-1
            )
            config_data["scrambled_like_random_graph_same_weights"] = scrambled_like_random_graph_same_weights_data

            # Same density Erdos-Renyi graph as the random order perturbed graph
            rop_like_random_graph = obtain_random_weighted_graph(
                num_nodes=random_order_perturbed_graph_data.metadata.num_nodes,
                required_unweighted_density=random_order_perturbed_graph_data.metadata.unweighted_density,
                required_weighted_density=random_order_perturbed_graph_data.metadata.weighted_density,
                seed=config.seed
            )
            rop_like_random_graph_data = GraphData(
                graph_obj=rop_like_random_graph,
                num_laplacian_paulis=-1
            )
            config_data["rop_like_random_graph"] = rop_like_random_graph_data

            # Same density Erdos-Renyi graph as the random order perturbed graph + SAME WEIGHTS DISTRIBUTION
            weights = np.abs(
                np.triu(random_order_perturbed_graph_data.laplacian_dense_matrix, k=1).flatten()
            )
            weights = weights[weights != 0]
            rop_like_random_graph_same_weights = obtain_random_weighted_graph(
                num_nodes=random_order_perturbed_graph_data.metadata.num_nodes,
                required_unweighted_density=random_order_perturbed_graph_data.metadata.unweighted_density,
                required_weighted_density=random_order_perturbed_graph_data.metadata.weighted_density,
                seed=config.seed,
                weights_distribution=weights
            )
            rop_like_random_graph_same_weights_data = GraphData(
                graph_obj=rop_like_random_graph_same_weights,
                num_laplacian_paulis=-1
            )
            config_data["rop_like_random_graph_same_weights"] = rop_like_random_graph_same_weights_data

            # Same density Erdos-Renyi graph as the definite order perturbed graph
            dop_like_random_graph = obtain_random_weighted_graph(
                num_nodes=definite_order_perturbed_graph_data.metadata.num_nodes,
                required_unweighted_density=definite_order_perturbed_graph_data.metadata.unweighted_density,
                required_weighted_density=definite_order_perturbed_graph_data.metadata.weighted_density,
                seed=config.seed
            )
            dop_like_random_graph_data = GraphData(
                graph_obj=dop_like_random_graph,
                num_laplacian_paulis=-1
            )
            config_data["dop_like_random_graph"] = dop_like_random_graph_data

            self.data.append(config_data)

    def analyze_results(self) -> None:
        """Analyze the spectral properties of all generated graphs.
        
        Computes eigenspectra for each graph and prepares data for similarity analysis.
        """
        for config_execution_result in self.data:

            comparison_spectra_pair = []

            for graph_type in GRAPH_TYPES:
                graph_data: GraphData = config_execution_result[graph_type]

                # Compute Laplacian spectrum
                graph_data.laplacian_spectrum = np.linalg.eigvalsh(graph_data.laplacian_dense_matrix)

                if graph_type in GRAPHS_COMPARISON_PAIR:
                    comparison_spectra_pair.append(graph_data.laplacian_spectrum)

            # Measure similarity of eigenspectrums
            config_execution_result["spectra_comparison"] = compare_hermitian_spectra(*comparison_spectra_pair)

    def save_results(self, data_dir_path: str | Path) -> None:
        """Save all experiment data and metadata to disk.
        
        Args:
            data_dir_path: Directory path where data will be saved.
        """
        self.data_dir_path = data_dir_path

        self.manifest_data = save_dataset(
            self.data,
            self.data_dir_path,
            experiment_metadata=self.metadata,
        )

    def plot_results(
        self,
        plot_window_start: float = 0.0,
        plot_window_ends: float = 1.0,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = OBSOLETE_GRAPHS,
        show_only: bool = False,
    ) -> None:
        """Plot the Laplacian spectra for each configuration.

        Args:
            plot_window_start: Start as a fraction of spectrum length (0.0-1.0).
            plot_window_ends: End as a fraction of spectrum length (0.0-1.0).
            merge_plots: If True, create a merged grid of all configurations.
            exclude_graphs: Graph types to skip in plots.
            show_only: If True, show the plots instead of saving them to disk.
        """

        # --- where to save ---
        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- plotting params ---
        default_markers = ['o', 's', '^', 'D', '+', 'x', 'P', '*']
        marker_cycler = cycle(default_markers)
        dpi = 300

        # --- figure grid if merge_plots ---
        configs_list = [res["configuration"] for res in self.data] if self.data else list(self.configurations)
        num_configs = len(self.data)  # safer than len(configs_list) if partially saved/loaded
        if merge_plots:
            num_rows = int(np.ceil(np.sqrt(num_configs))) or 1
            num_cols = int(np.ceil(num_configs / num_rows)) or 1
            merged_fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
            if num_configs == 1:
                axes = np.array([[axes]])
            elif num_rows == 1:
                axes = axes.reshape(1, -1)

        # --- iterate configs + results + manifest items ---
        for idx, (config, config_execution_result, manifest_item) in enumerate(
            zip(configs_list, self.data, self.manifest_data["items"])
        ):
            num_nodes = config.num_nodes
            
            window_start = max(0, min(int(plot_window_start * num_nodes), num_nodes))
            window_ends = max(window_start, min(int(plot_window_ends * num_nodes), num_nodes))
            nodes_indexes = np.arange(num_nodes)
            shown = max(1, window_ends - window_start)
            scatter_size = max(100 / shown, 1.0)

            # always make individual plot
            plt.figure()

            ax = None
            if merge_plots:
                row_idx = idx // num_cols
                col_idx = idx % num_cols
                ax = axes[row_idx, col_idx]

            for graph_type in GRAPH_TYPES:
                if graph_type in exclude_graphs:
                    continue
                if graph_type not in config_execution_result:
                    continue
                if graph_type not in GRAPHS_TO_PLOT_MAP:
                    continue

                bundle: GraphData = config_execution_result[graph_type]
                spec = bundle.laplacian_spectrum
                if spec is None:
                    continue

                # get a marker for this graph type (stable index if possible)
                try:
                    marker = default_markers[GRAPH_TYPES.index(graph_type) % len(default_markers)]
                except Exception:
                    marker = next(marker_cycler)

                graph_label_name = GRAPHS_TO_PLOT_MAP[graph_type]
                label = (
                    f"{graph_label_name} ($|E| = $ {bundle.metadata.num_edges}, "
                    f"$|P| = ${bundle.num_laplacian_paulis}, "
                    f"$|P_G| = ${bundle.num_commuting_groups})"
                )
                
                plt.scatter(
                    nodes_indexes[window_start:window_ends],
                    np.asarray(spec)[window_start:window_ends],
                    s=scatter_size,
                    label=label,
                    marker=marker,
                )

                if ax is not None:
                    ax.scatter(
                        nodes_indexes[window_start:window_ends],
                        np.asarray(spec)[window_start:window_ends],
                        s=scatter_size,
                        label=label,
                        marker=marker,
                    )

            title_str = (
                f"Config: {config}\n"
                f"{config_execution_result['spectra_comparison']}"
            )

            plt.xlabel("Eigenvalue index")
            plt.ylabel("Eigenvalue")
            plt.legend(
                loc="upper left",
                frameon=True,
                fontsize=8,
                markerscale=2,
                framealpha=0.9
            )
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.title(title_str, fontsize=8)
            
            if not show_only:
                out_png = Path(run_dir, manifest_item["item_id"], "spectra_plot.png")
                out_png.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
                plt.close()

            # format merged figure
            if ax is not None:
                ax.set_xlabel("Eigenvalue index")
                ax.set_ylabel("Eigenvalue")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.set_title(title_str, fontsize=8)
                ax.legend(
                    loc="upper left",
                    frameon=True,
                    fontsize=7,
                    markerscale=2,
                    framealpha=0.9
                )

        if merge_plots:
            # prune any unused axes
            total_axes = axes.size
            for j in range(num_configs, total_axes):
                r = j // num_cols
                c = j % num_cols
                merged_fig.delaxes(axes[r, c])

            merged_fig.tight_layout()
            if not show_only:
                merged_path = Path(run_dir, "merged_spectra_plot.png")
                merged_fig.savefig(merged_path, dpi=dpi, bbox_inches="tight")
                plt.close(merged_fig)

        if show_only:
            plt.show()

    def plot_matrices(
        self,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = OBSOLETE_GRAPHS,
        show_only: bool = False,
    ) -> None:
        """Create sparsity pattern visualizations of Laplacian matrices.
        
        Args:
            merge_plots: If True, create a merged grid of all configurations per graph type.
            exclude_graphs: Graph types to skip in visualization.
            show_only: If True, show the plots instead of saving them to disk.
        """
        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        configs_list = [res["configuration"] for res in self.data] if self.data else list(self.configurations)
        num_configs = len(self.data)
        
        merged_figs = {}
        merged_axes = {}
        if merge_plots:
            num_rows = int(np.ceil(np.sqrt(num_configs))) or 1
            num_cols = int(np.ceil(num_configs / num_rows)) or 1
            
            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs:
                    continue
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))
                if num_configs == 1:
                    axes = np.array([[axes]])
                elif num_rows == 1:
                    axes = axes.reshape(1, -1)
                merged_figs[graph_name] = fig
                merged_axes[graph_name] = axes

        for idx, config_data in enumerate(self.data):
            config = configs_list[idx]
            config_data_path = Path(
                run_dir,
                self.manifest_data["items"][config_data["config_index"]]["item_id"]
            )
            config_data_path.mkdir(parents=True, exist_ok=True)

            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs:
                    continue
                if graph_name not in config_data:
                    continue
            
                # Individual plot
                plt.figure()
                plt.spy(config_data[graph_name].laplacian_dense_matrix, markersize=0.1)
                plt.title(f"{graph_name} - {config}")
                if not show_only:
                    plt.savefig(Path(config_data_path, f"{graph_name}_laplacian.png"))
                    plt.close()

                # Merged plot
                if merge_plots and graph_name in merged_axes:
                    row_idx = idx // num_cols
                    col_idx = idx % num_cols
                    ax = merged_axes[graph_name][row_idx, col_idx]
                    
                    L = config_data[graph_name].laplacian_dense_matrix
                    ax.spy(L, markersize=0.2)
                    ax.set_title(f"Config: {config}, nonzero_rate = {np.count_nonzero(L) / L.size:.2f}", fontsize=8)

        if merge_plots:
            for graph_name, fig in merged_figs.items():
                axes = merged_axes[graph_name]
                total_axes = axes.size
                for j in range(num_configs, total_axes):
                    r = j // num_cols
                    c = j % num_cols
                    fig.delaxes(axes[r, c])
                
                fig.tight_layout()
                if not show_only:
                    merged_path = Path(run_dir, f"merged_{graph_name}_matrices.png")
                    fig.savefig(merged_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

        if show_only:
            plt.show()

    def draw_graphs(
        self,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = OBSOLETE_GRAPHS,
        show_only: bool = False,
    ) -> None:
        """Draw the graph structures using NetworkX.
        
        Args:
            merge_plots: If True, create a merged grid of all configurations per graph type.
            exclude_graphs: Graph types to skip in visualization.
            show_only: If True, show the plots instead of saving them to disk.
        """
        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        configs_list = [res["configuration"] for res in self.data] if self.data else list(self.configurations)
        num_configs = len(self.data)
        
        merged_figs = {}
        merged_axes = {}
        if merge_plots:
            num_rows = int(np.ceil(np.sqrt(num_configs))) or 1
            num_cols = int(np.ceil(num_configs / num_rows)) or 1
            
            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs:
                    continue
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
                if num_configs == 1:
                    axes = np.array([[axes]])
                elif num_rows == 1:
                    axes = axes.reshape(1, -1)
                merged_figs[graph_name] = fig
                merged_axes[graph_name] = axes

        for idx, config_data in enumerate(self.data):
            config = configs_list[idx]
            config_data_path = Path(
                run_dir,
                self.manifest_data["items"][config_data["config_index"]]["item_id"]
            )
            config_data_path.mkdir(parents=True, exist_ok=True)

            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs:
                    continue
                if graph_name not in config_data:
                    continue
                
                graph_obj = config_data[graph_name].graph_obj
                if graph_obj is None:
                    continue

                # Prepare layout and aesthetic features
                pos = nx.spring_layout(graph_obj, seed=42)
                degrees = [deg for node, deg in graph_obj.degree(weight="weight")]
                num_edges = graph_obj.number_of_edges()
                edge_alpha = max(0.05, min(0.5, 3.0 / np.sqrt(max(num_edges, 1))))
                
                # Individual plot
                plt.figure(figsize=(8, 6))
                ax = plt.gca()
                nx.draw_networkx_nodes(
                    graph_obj, pos, ax=ax,
                    node_size=25,
                    node_color=degrees,
                    cmap=plt.cm.plasma,
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.5
                )
                nx.draw_networkx_edges(
                    graph_obj, pos, ax=ax,
                    alpha=edge_alpha,
                    edge_color="gray",
                    width=0.8
                )
                ax.axis("off")
                plt.title(f"{graph_name} - {config}")
                
                if not show_only:
                    plt.savefig(Path(config_data_path, f"{graph_name}_graph.png"), dpi=300, bbox_inches="tight")
                    plt.close()

                # Merged plot
                if merge_plots and graph_name in merged_axes:
                    row_idx = idx // num_cols
                    col_idx = idx % num_cols
                    ax = merged_axes[graph_name][row_idx, col_idx]
                    
                    nx.draw_networkx_nodes(
                        graph_obj, pos, ax=ax,
                        node_size=15,
                        node_color=degrees,
                        cmap=plt.cm.plasma,
                        alpha=0.9,
                        edgecolors="white",
                        linewidths=0.3
                    )
                    nx.draw_networkx_edges(
                        graph_obj, pos, ax=ax,
                        alpha=edge_alpha,
                        edge_color="gray",
                        width=0.5
                    )
                    ax.axis("off")
                    ax.set_title(f"Config: {config}", fontsize=9)

        if merge_plots:
            for graph_name, fig in merged_figs.items():
                axes = merged_axes[graph_name]
                total_axes = axes.size
                for j in range(num_configs, total_axes):
                    r = j // num_cols
                    c = j % num_cols
                    fig.delaxes(axes[r, c])
                
                fig.tight_layout()
                if not show_only:
                    merged_path = Path(run_dir, f"merged_{graph_name}_graphs.png")
                    fig.savefig(merged_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

        if show_only:
            plt.show()

    def run_all(self, filepath: str, draw_graphs: bool = False) -> None:
        """Execute the complete experiment pipeline.
        
        Args:
            filepath: Path where results will be saved.
        """

        self.perform_experiment()
        self.analyze_results()
        self.save_results(filepath)
        self.plot_results()
        self.plot_matrices()

        if draw_graphs:
            self.draw_graphs()


if __name__ == "__main__":

    ec = ExperimentConfigurations(
        n_num_qubits=[10], # q
        d_skeleton_regularity=[3],
        max_skeleton_locality=[3],
        num_perturbations=[
            # 0,
            # lambda x: int(np.sqrt(x)),
            # lambda x: x,
            lambda x: x**2,
            # lambda x: 2**x,
            # lambda x: x**3,
            lambda x: 2 * x**3,
        ],
        max_perturbation_locality=[3, 4, 5, 6, 7], # m
        perturbation_weights_bounds=[(0.5, 5)],
        seed=[32],
    )

    experiment = LaplacianHamiltoniansWorkshop(configurations=ec)
    experiment.run_all("experiments_data_archive", draw_graphs=False)

    # data_dir_path = Path(
    #     "/home/ohad-lev/ohad/msc/research/thesis/qsga/experiments_data_archive"
    # )
    # experiment = LaplacianHamiltoniansWorkshop.from_data(Path(data_dir_path, "2026-05-18_09-35-31"))
    # experiment.analyze_results()
    # experiment.plot_results()#show_only=True)
    # # experiment.plot_matrices()#show_only=True)
    # # experiment.draw_graphs()#show_only=True)