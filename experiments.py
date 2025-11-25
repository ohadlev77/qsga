from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Iterator, Iterable, TYPE_CHECKING

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from hamiltonian_generators import obtain_ix_n_hamiltonian, obtain_random_perturbated_laplacian
from data_verifiers import is_valid_laplacian
from util import obtain_random_weighted_graph, compute_weighted_density, transform_laplacian_to_graph
from data_handling import save_dataset, load_dataset, _slugify, GRAPH_TYPES

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SingleExperimentConfiguration:
    """TODO COMPLETE."""

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
        base = (
            f"n{self.n_num_qubits}-"
            f"d{self.d_skeleton_regularity}-"
            f"sl{self.max_skeleton_locality}-"
            f"p{self.num_perturbations}-"
            f"pl{self.max_perturbation_locality}"
        )

        if self.seed is not None:
            base += f"-s{self.seed}"
            
        return _slugify(base)


@dataclass(frozen=True)
class ExperimentConfigurations:
    """TODO COMPLETE."""

    n_num_qubits: list[int]
    d_skeleton_regularity: list[int]
    max_skeleton_locality: list[int]
    num_perturbations: list[int]
    max_perturbation_locality: list[int]

    perturbation_weights_bounds: list[tuple[float, float] | None] | None = None
    seed: list[int] | None = None

    def __iter__(self) -> Iterator[SingleExperimentConfiguration]:
        """TODO COMPLETE"""

        for vals in product(
            self.n_num_qubits,
            self.d_skeleton_regularity,
            self.max_skeleton_locality,
            self.num_perturbations,
            self.max_perturbation_locality,
            self.perturbation_weights_bounds or [None],
            self.seed or [None],
        ):
            yield SingleExperimentConfiguration(*vals)


@dataclass
class GraphMetadata:
    num_nodes: int
    num_edges: int
    unweighted_density: float
    weighted_density: float


@dataclass
class GraphData:
    laplacian_sparse_obj: SparsePauliOp | None = None
    graph_obj: nx.Graph | None = None

    def __post_init__(self) -> None:
        if self.laplacian_sparse_obj is not None:
            is_valid_laplacian(self.laplacian_sparse_obj, throw_exception=True)
            self.laplacian_pauli_repr: list[tuple[str, complex]] = self.laplacian_sparse_obj.to_list()
            self.laplacian_sparse_pauli_repr: list[tuple[str, list[int], complex]] = self.laplacian_sparse_obj.to_sparse_list()

        if self.graph_obj is None:
            self.graph_obj: nx.Graph = transform_laplacian_to_graph(self.laplacian_sparse_obj)
        self.laplacian_dense_matrix: NDArray[np.float64] = nx.laplacian_matrix(self.graph_obj).todense()
        
        self.metadata: GraphMetadata[int | float] = GraphMetadata(
            num_nodes=self.graph_obj.number_of_nodes(),
            num_edges=self.graph_obj.number_of_edges(),
            unweighted_density=nx.density(self.graph_obj),
            weighted_density=compute_weighted_density(self.graph_obj)
        )


class LaplacianHamiltoniansGeneration:
    """TODO COMPLET."""

    @staticmethod
    def from_data(data_dir_path: Path | str) -> LaplacianHamiltoniansGeneration:
        """Load both dataset and configurations from a saved run."""
        
        data, manifest_data, metadata = load_dataset(data_dir_path)

        configurations = ExperimentConfigurations(**metadata["configurations"])

        obj = LaplacianHamiltoniansGeneration.__new__(LaplacianHamiltoniansGeneration)
        obj.data = data
        obj.configurations = configurations
        obj.manifest_data = manifest_data
        obj.metadata = metadata

        return obj

    def __init__(self, configurations: ExperimentConfigurations) -> None:
        """TODO COMPLETE."""

        self.configurations = configurations
        self.data = []
        self.metadata = {
            "configurations": asdict(configurations),
            "total_configurations": len(list(configurations)),
            "graph_types": GRAPH_TYPES
        }

    def perform_experiment(self) -> None:
        """TODO COMPLETE."""

        for config_index, config in enumerate(self.configurations):

            # Skeleton Laplacian
            skeleton_laplacian = obtain_ix_n_hamiltonian(
                n=config.n_num_qubits,
                d=config.d_skeleton_regularity,
                max_locality=config.max_skeleton_locality,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            skeleton_graph_data = GraphData(laplacian_sparse_obj=skeleton_laplacian)

            # Perturbated Laplacians
            kwargs = dict(
                skeleton_hamiltonian=skeleton_laplacian,
                num_perturbations=config.num_perturbations,
                max_perturbation_locality=config.max_perturbation_locality,
                random_perturbation_weights_bounds=config.perturbation_weights_bounds,
            )
            
            definite_order_perturbated_laplacian = obtain_random_perturbated_laplacian(
                **kwargs,
                random_perturbations_scaling=False,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            definite_order_perturbated_graph_data = GraphData(laplacian_sparse_obj=definite_order_perturbated_laplacian)
            
            random_order_perturbated_laplacian = obtain_random_perturbated_laplacian(
                **kwargs,
                random_perturbations_scaling=True,
                pseudo_rng=np.random.default_rng(seed=config.seed)
            )
            random_order_perturbated_graph_data = GraphData(laplacian_sparse_obj=random_order_perturbated_laplacian)

            config_data = {
                "config_index": config_index,
                "configuration": config,
                "skeleton_graph": skeleton_graph_data,
                "definite_order_perturbated_graph": definite_order_perturbated_graph_data,
                "random_order_perturbated_graph": random_order_perturbated_graph_data,
            }

            # Same density Erdos-Renyi graph as the random order perturbated graph
            rop_like_random_graph = obtain_random_weighted_graph(
                num_nodes=random_order_perturbated_graph_data.metadata.num_nodes,
                required_unweighted_density=random_order_perturbated_graph_data.metadata.unweighted_density,
                required_weighted_density=random_order_perturbated_graph_data.metadata.weighted_density,
                seed=config.seed
            )
            rop_like_random_graph_data = GraphData(graph_obj=rop_like_random_graph)
            config_data["rop_like_random_graph"] = rop_like_random_graph_data

            # Same density Erdos-Renyi graph as the definite order perturbated graph
            dop_like_random_graph = obtain_random_weighted_graph(
                num_nodes=definite_order_perturbated_graph_data.metadata.num_nodes,
                required_unweighted_density=definite_order_perturbated_graph_data.metadata.unweighted_density,
                required_weighted_density=definite_order_perturbated_graph_data.metadata.weighted_density,
                seed=config.seed
            )
            dop_like_random_graph_data = GraphData(graph_obj=dop_like_random_graph)
            config_data["dop_like_random_graph"] = dop_like_random_graph_data

            self.data.append(config_data)

    def analyze_results(self) -> None:
        """TODO COMPLETE."""

        # Compute eigenspectrums
        for config_execution_result in self.data:
            for graph_type in GRAPH_TYPES:
                graph_data: GraphData = config_execution_result[graph_type]
                graph_data.laplacian_spectrum = np.linalg.eigvalsh(graph_data.laplacian_dense_matrix) # TODO laplacian_spectrum define somewhere

        # Measure similarity of eigenspectrums - TODO COMPLETE
        pass 

    def save_results(self, data_dir_path: str | Path) -> None:
        self.manifest_data = save_dataset(
            self.data,
            data_dir_path,
            experiment_metadata=self.metadata,
        )

    def plot_results(
        self,
        plot_window_start: float = 0.0,
        plot_window_ends: float = 1.0,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = ("definite_order_perturbated_graph", "dop_like_random_graph"),
    ) -> None:
        """
        *** THIS METHOD IS VIBE-CODED *** 
        Plot the Laplacian spectra for each configuration.

        Args:
            plot_window_start: start as a fraction of the spectrum length (0.0-1.0).
            plot_window_ends: end as a fraction of the spectrum length (0.0-1.0).
            merge_plots: if True, also create a merged grid of all configurations.
            exclude_graphs: graph types to skip in plots.
        """
        from itertools import cycle

        # --- helpers ---
        def _get_spectrum(bundle):
            # Works for both GraphData objects and dicts
            if isinstance(bundle, dict):
                return bundle.get("laplacian_spectrum")
            return getattr(bundle, "laplacian_spectrum", None)

        def _get_num_nodes(bundle):
            # Try to infer number of nodes from Laplacian shape; else from graph
            lap = None
            if isinstance(bundle, dict):
                lap = bundle.get("laplacian_obj")
                g = bundle.get("graph_obj")
            else:
                lap = getattr(bundle, "laplacian_dense_matrix", None) or getattr(bundle, "laplacian_obj", None)
                g = getattr(bundle, "graph_obj", None)
            if lap is not None:
                return int(np.asarray(lap).shape[0])
            if g is not None:
                return int(g.number_of_nodes())
            return None

        # --- where to save ---
        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- plotting params ---
        default_markers = ['o', 's', '^', 'D', '+', 'x', 'P', '*']
        marker_cycler = cycle(default_markers)
        dpi = 300

        # --- figure grid if merge_plots ---
        configs_list = list(self.configurations)  # materialize once (preserves order of self.data)
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
            # pick a representative bundle to determine spectrum length
            # prefer "skeleton_graph", else any available
            ref_bundle_name = next((g for g in GRAPH_TYPES if g in config_execution_result), None)
            if ref_bundle_name is None:
                continue
            ref_bundle = config_execution_result[ref_bundle_name]

            # num_nodes and windowing
            num_nodes = getattr(config, "num_nodes", None) or _get_num_nodes(ref_bundle) or 0
            if num_nodes <= 0:
                continue
            window_start = max(0, min(int(plot_window_start * num_nodes), num_nodes))
            window_ends = max(window_start, min(int(plot_window_ends * num_nodes), num_nodes))
            nodes_indexes = np.arange(num_nodes)
            shown = max(1, window_ends - window_start)
            scatter_size = max(100 / shown, 1.0)

            # always make individual plot
            plt.figure()
            for graph_type in GRAPH_TYPES:
                if graph_type in exclude_graphs:
                    continue
                if graph_type not in config_execution_result:
                    continue

                bundle = config_execution_result[graph_type]
                spec = _get_spectrum(bundle)
                if spec is None:
                    continue

                # get a marker for this graph type (stable index if possible)
                try:
                    marker = default_markers[GRAPH_TYPES.index(graph_type) % len(default_markers)]
                except Exception:
                    marker = next(marker_cycler)

                plt.scatter(
                    nodes_indexes[window_start:window_ends],
                    np.asarray(spec)[window_start:window_ends],
                    s=scatter_size,
                    label=graph_type,
                    marker=marker,
                )

            plt.xlabel("Eigenvalue index")
            plt.ylabel("Eigenvalue")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.title(f"Spectrum plot for configuration: {config}")
            out_png = Path(run_dir, manifest_item["item_id"], "spectra_plot.png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.close()

            # add to merged figure
            if merge_plots:
                row_idx = idx // num_cols
                col_idx = idx % num_cols
                ax = axes[row_idx, col_idx]

                for graph_type in GRAPH_TYPES:
                    if graph_type in exclude_graphs:
                        continue
                    if graph_type not in config_execution_result:
                        continue

                    bundle = config_execution_result[graph_type]
                    spec = _get_spectrum(bundle)
                    if spec is None:
                        continue

                    try:
                        marker = default_markers[GRAPH_TYPES.index(graph_type) % len(default_markers)]
                    except Exception:
                        marker = next(marker_cycler)

                    ax.scatter(
                        nodes_indexes[window_start:window_ends],
                        np.asarray(spec)[window_start:window_ends],
                        s=scatter_size,
                        label=graph_type,
                        marker=marker,
                    )

                ax.set_xlabel("Eigenvalue index")
                ax.set_ylabel("Eigenvalue")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.set_title(f"Config: {config}", fontsize=8)
                # avoid duplicate legends: show once per axis (fine)
                ax.legend(fontsize=8)

        if merge_plots:
            # prune any unused axes
            total_axes = axes.size
            for j in range(num_configs, total_axes):
                r = j // num_cols
                c = j % num_cols
                merged_fig.delaxes(axes[r, c])

            merged_fig.tight_layout()
            merged_path = Path(run_dir, "merged_spectra_plot.png")
            merged_fig.savefig(merged_path, dpi=dpi, bbox_inches="tight")
            plt.close(merged_fig)

    def run_all(self, filepath: str) -> None:
        """TODO THIS METHOD IS DUMMY FOR NOW."""

        self.perform_experiment()
        self.analyze_results()
        self.save_results(filepath)
        self.plot_results()


if __name__ == "__main__":
    ec = ExperimentConfigurations(
        n_num_qubits=[5, 6],
        d_skeleton_regularity=[3, 4],
        max_skeleton_locality=[3],
        num_perturbations=[6],
        max_perturbation_locality=[3],
        perturbation_weights_bounds=[(2, 80)],
        seed=[32],
    )

    # ec = ExperimentConfigurations(
    #     n_num_qubits=[6],
    #     d_skeleton_regularity=[7],
    #     max_skeleton_locality=[3],
    #     num_perturbations=[9],
    #     max_perturbation_locality=[3],
    #     perturbation_weights_bounds=[(2, 6)],
    #     seed=[32],
    # )

    experiment = LaplacianHamiltoniansGeneration(configurations=ec)
    experiment.perform_experiment()
    experiment.analyze_results()
    experiment.save_results("experiments_data_archive")
    
    experiment.plot_results()

    # rexp = LaplacianHamiltoniansGeneration.from_data(
    #     "/home/ohad-lev/ohad/msc/research/thesis/qsga/experiments_data_archive/2025-11-05_14-49-35"
    # )
    # rexp.plot_results(plot_window_ends=10)