from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product, cycle
from pathlib import Path
from typing import Iterator, Iterable, TYPE_CHECKING, Callable, Any

import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from qsga import (
    GRAPH_TYPES,
    OBSOLETE_GRAPHS,
    GRAPHS_TO_PLOT_MAP,
    GRAPHS_COMPARISON_PAIR
)
from qsga.data_verifiers import is_valid_laplacian
from qsga.data_handling import (
    IncrementalSaver,
    save_dataset,
    load_dataset,
    _slugify,
    _config_to_jsonable,
    _derive_item_slug
)
from qsga.util import (
    obtain_random_weighted_graph,
    compute_weighted_density,
    transform_laplacian_to_graph,
    compare_hermitian_spectra,
    time_it
)
from qsga.hamiltonian_generators import (
    obtain_skeleton_laplacian,
    obtain_random_perturbed_laplacian,
    PerturbationScalingMethod
)
from qsga.logging_setup import logger, configure_file_logging, reset_log_capture, write_timing_report

if TYPE_CHECKING:
    from numpy.typing import NDArray


FIGURES_DPI = 300


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

    @time_it(last=True)
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

        # self.num_laplacian_paulis == -1 => A graph is given, not a Laplacian operator
        op = self.laplacian_sparse_obj
        if self.num_laplacian_paulis == -1 and self.laplacian_sparse_obj is None:
            self.num_laplacian_paulis = "N/A"
            
            if self.metadata.num_nodes <= 2**13:
                op = SparsePauliOp.from_operator(self.laplacian_dense_matrix).simplify()
                self.num_laplacian_paulis = len(op)

        if op is not None:
            self.num_commuting_groups = "N/A"

            if isinstance(self.num_laplacian_paulis, int):
                if self.num_laplacian_paulis < 10_000:
                    self.num_commuting_groups = len(op.group_commuting())

        logger.info(self.__str__())

    def __str__(self) -> str:
        """
        Generate a string representation of the GraphData instance.
        Shows key graph metrics: nodes, edges, density, weighted density, 
        and Hamiltonian terms if available.

        Returns:
            str: A formatted string summarizing the graph's key attributes.
        """
        parts = [
            f"nodes={self.metadata.num_nodes}",
            f"edges={self.metadata.num_edges}",
            f"density={self.metadata.unweighted_density:.4f}",
            f"weighted_density={self.metadata.weighted_density:.4f}",
            f"paulis={self.num_laplacian_paulis}",
            f"commuting_groups={self.num_commuting_groups}"
        ]
        return f"GraphData({', '.join(parts)})"


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

                    if "num_laplacian_paulis" in b_dict:
                        gd.num_laplacian_paulis = b_dict["num_laplacian_paulis"]
                    elif "laplacian_pauli_repr" in b_dict:
                        gd.num_laplacian_paulis = len(b_dict["laplacian_pauli_repr"])
                    else:
                        gd.num_laplacian_paulis = "N/A"

                    gd.num_commuting_groups = b_dict.get("num_commuting_groups", "N/A")
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

    def perform_experiment(self, save_dir: str | Path | None = None) -> None:
        """Generate all Laplacian graphs and compute their properties.

        For each configuration, generates:
        - Skeleton Laplacian graph.
        - Randomly Perturbed Scrambled Laplacian (`PerturbationScalingMethod.SCRAMBLE`).
        - Erdős–Rényi graphs matching the scrambled and ROP perturbed graphs in density and weights.
        """
        reset_log_capture()

        saver = None
        if save_dir is not None:
            saver = IncrementalSaver(
                out_dir=save_dir,
                experiment_metadata=self.metadata,
            )
            self.data_dir_path = save_dir
            self.metadata = saver.experiment_metadata
            configure_file_logging(saver.run_dir / "run.log")

        try:
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
                    "random_order_scrambled_perturbed_graph": random_order_scrambled_perturbed_graph_data,
                }

                # Erdős–Rényi graph matching scrambled perturbed graph density + weights
                weights = np.abs(
                    np.triu(random_order_scrambled_perturbed_graph_data.laplacian_dense_matrix, k=1).flatten()
                )
                weights = weights[weights != 0]
                config_data["scrambled_like_random_graph_same_weights"] = GraphData(
                    graph_obj=obtain_random_weighted_graph(
                        num_nodes=random_order_scrambled_perturbed_graph_data.metadata.num_nodes,
                        required_unweighted_density=random_order_scrambled_perturbed_graph_data.metadata.unweighted_density,
                        required_weighted_density=random_order_scrambled_perturbed_graph_data.metadata.weighted_density,
                        seed=config.seed,
                        weights_distribution=weights
                    ),
                    num_laplacian_paulis=-1
                )

                # Analyze configuration on-the-fly
                logger.info(f"[config {config_index}] Starting spectral analysis ({config.num_nodes} nodes, {len(GRAPH_TYPES)} graph types)...")
                self._analyze_single_configuration(config_data)

                self.data.append(config_data)

                if saver is not None:
                    logger.info(f"[config {config_index}] Saving data to disk...")
                    saver.save_item(config_index, config_data)
                    self.manifest_data = {"items": saver.manifest_items}

        except Exception as e:
            logger.exception("An error occurred during the experiment execution loop.")
            raise e
        finally:
            if saver is not None:
                write_timing_report(saver.run_dir / "timing_report.txt")

    @time_it(last=True)
    def _analyze_single_configuration(self, config_execution_result: dict[str, Any]) -> None:
        """Compute eigenspectra and compare them for a single configuration."""
        comparison_spectra_pair = []

        graphs_needing_spectrum = set(GRAPHS_COMPARISON_PAIR) | set(GRAPHS_TO_PLOT_MAP.keys())

        for graph_type in GRAPH_TYPES:
            if graph_type not in graphs_needing_spectrum:
                continue

            graph_data: GraphData = config_execution_result[graph_type]

            # Compute Laplacian spectrum if not already present
            if graph_data.laplacian_spectrum is None:
                n = graph_data.laplacian_dense_matrix.shape[0]
                logger.info(f"[eigvalsh] {graph_type}: computing spectrum ({n}x{n} matrix)...")
                graph_data.laplacian_spectrum = time_it(np.linalg.eigvalsh)(graph_data.laplacian_dense_matrix)
            else:
                logger.info(f"[eigvalsh] {graph_type}: spectrum already present, skipping")

            if graph_type in GRAPHS_COMPARISON_PAIR:
                comparison_spectra_pair.append(graph_data.laplacian_spectrum)

        # Measure similarity of eigenspectrums
        logger.info("Computing spectra comparison (Soergel distance)...")
        config_execution_result["spectra_comparison"] = time_it(compare_hermitian_spectra)(*comparison_spectra_pair)
        logger.info(f"Spectra comparison result: {config_execution_result['spectra_comparison']}")

    def analyze_results(self) -> None:
        """Analyze the spectral properties of all generated graphs.
        
        Computes eigenspectra for each graph and prepares data for similarity analysis.
        """
        for config_execution_result in self.data:
            if "spectra_comparison" in config_execution_result:
                continue
            self._analyze_single_configuration(config_execution_result)

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

        run_dir = Path(self.metadata["run_metadata"]["run_dir"])
        configure_file_logging(run_dir / "run.log")
        write_timing_report(run_dir / "timing_report.txt")


    def _get_scanned_params_from_config(self) -> list[str]:
        """Return field names that have >1 unique value in the original ExperimentConfigurations.

        Uses the original config lists rather than resolved SingleExperimentConfiguration values
        so that lambda-derived params (e.g. num_perturbations=lambda x: x**2) are not mistakenly
        treated as independently scanned dimensions.
        """
        fields = [
            "n_num_qubits", "d_skeleton_regularity", "max_skeleton_locality",
            "num_perturbations", "max_perturbation_locality", "perturbation_weights_bounds", "seed"
        ]
        result = []
        for field in fields:
            vals = getattr(self.configurations, field, None)
            if vals is not None and len(vals) > 1:
                result.append(field)
        return result

    def _create_merged_figure(self, configs_list, base_figsize, scanned_params_order=None, orientation="horizontal", restrict_params=None):

        fields = [
            "n_num_qubits", "d_skeleton_regularity", "max_skeleton_locality",
            "num_perturbations", "max_perturbation_locality", "perturbation_weights_bounds", "seed"
        ]

        unique_values = {}
        for field in fields:
            seen = []
            for cfg in configs_list:
                val = getattr(cfg, field)
                if val not in seen:
                    seen.append(val)
            unique_values[field] = seen

        scanned_params = [f for f in fields if len(unique_values[f]) > 1]

        # Drop params that weren't independently varied in the original config spec
        # (e.g. num_perturbations derived via a lambda from n_num_qubits)
        if restrict_params is not None:
            scanned_params = [p for p in scanned_params if p in restrict_params]

        if scanned_params_order is not None:
            ordered = [p for p in scanned_params_order if p in scanned_params]
            ordered.extend([p for p in scanned_params if p not in ordered])
            scanned_params = ordered
        else:
            scanned_params.sort(key=lambda p: len(unique_values[p]), reverse=True)

        if len(scanned_params) == 0:
            fig, axes = plt.subplots(1, 1, figsize=base_figsize, layout='constrained')
            return fig, {(): axes}, scanned_params, unique_values

        elif len(scanned_params) == 1:
            p0_len = len(unique_values[scanned_params[0]])
            fig, axes = plt.subplots(1, p0_len, figsize=(base_figsize[0]*p0_len, base_figsize[1]), layout='constrained')
            if p0_len == 1:
                axes = [axes]
            return fig, {(i,): axes[i] for i in range(p0_len)}, scanned_params, unique_values

        elif len(scanned_params) == 2:
            c_len = len(unique_values[scanned_params[0]])
            r_len = len(unique_values[scanned_params[1]])
            fig, axes = plt.subplots(r_len, c_len, figsize=(base_figsize[0]*c_len, base_figsize[1]*r_len), layout='constrained')
            if r_len == 1 and c_len == 1:
                axes = np.array([[axes]])
            elif r_len == 1:
                axes = np.array([axes])
            elif c_len == 1:
                axes = np.array([[a] for a in axes])
            return fig, {(c, r): axes[r, c] for r in range(r_len) for c in range(c_len)}, scanned_params, unique_values

        else:
            # --- 3D+ Unified Flat GridSpec (Packed Outer Groups) ---
            inner_p0 = scanned_params[0]
            inner_p1 = scanned_params[1]
            outer_params = scanned_params[2:]

            c_len = len(unique_values[inner_p0])
            r_len = len(unique_values[inner_p1])

            # Find actual existing outer groups to avoid cartesian "holes"
            outer_groups = []
            for cfg in configs_list:
                group_key = tuple(getattr(cfg, p) for p in outer_params)
                if group_key not in outer_groups:
                    outer_groups.append(group_key)

            num_subfigs = len(outer_groups)

            if orientation == "horizontal":
                sf_cols = int(np.ceil(np.sqrt(num_subfigs)))
                sf_rows = int(np.ceil(num_subfigs / sf_cols))
            else:
                sf_rows = int(np.ceil(np.sqrt(num_subfigs)))
                sf_cols = int(np.ceil(num_subfigs / sf_rows))

            TITLE_H = 0.15
            SPACER_H = 0.3
            SPACER_W = 0.2

            width_ratios = []
            for c in range(sf_cols):
                width_ratios.extend([1.0] * c_len)
                if c < sf_cols - 1:
                    width_ratios.append(SPACER_W)

            height_ratios = []
            for r in range(sf_rows):
                height_ratios.append(TITLE_H)
                height_ratios.extend([1.0] * r_len)
                if r < sf_rows - 1:
                    height_ratios.append(SPACER_H)

            total_c = len(width_ratios)
            total_r = len(height_ratios)

            fig_width = base_figsize[0] * c_len * sf_cols + (sf_cols - 1) * (base_figsize[0] * SPACER_W)
            fig_height = base_figsize[1] * r_len * sf_rows + sf_rows * (base_figsize[1] * TITLE_H) + (sf_rows - 1) * (base_figsize[1] * SPACER_H)

            fig = plt.figure(figsize=(fig_width, fig_height), layout='constrained', dpi=FIGURES_DPI)
            gs = GridSpec(total_r, total_c, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)

            axes_map = {}
            title_axes_drawn = set()

            # Map the plots based ONLY on existing configurations
            for cfg in configs_list:
                # 1. Determine which tightly-packed block this belongs to
                outer_key = tuple(getattr(cfg, p) for p in outer_params)
                sf_idx = outer_groups.index(outer_key)

                sf_r = sf_idx // sf_cols
                sf_c = sf_idx % sf_cols

                # 2. Determine inner coordinates
                inner_c = unique_values[inner_p0].index(getattr(cfg, inner_p0))
                inner_r = unique_values[inner_p1].index(getattr(cfg, inner_p1))

                # 3. Create full coord key for plot_results to match
                coord = tuple(unique_values[p].index(getattr(cfg, p)) for p in scanned_params)

                # 4. Map to global grid
                group_start_row = sf_r * (r_len + 2)
                group_start_col = sf_c * (c_len + 1)

                # Render overarching title once per block
                if sf_idx not in title_axes_drawn:
                    title_ax = fig.add_subplot(gs[group_start_row, group_start_col : group_start_col + c_len])
                    title_ax.axis('off')
                    
                    title_parts = [f"{p}={v}" for p, v in zip(outer_params, outer_key)]
                    title_ax.text(0.5, 0.5, " | ".join(title_parts),
                                  ha='center', va='center',
                                  fontsize=16, fontweight='bold',
                                  transform=title_ax.transAxes)
                    title_axes_drawn.add(sf_idx)

                # Render plot axis
                if coord not in axes_map:
                    ax = fig.add_subplot(gs[group_start_row + 1 + inner_r, group_start_col + inner_c])
                    axes_map[coord] = ax

            return fig, axes_map, scanned_params, unique_values

    def plot_results(
        self,
        plot_window_start: float = 0.0,
        plot_window_ends: float = 1.0,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = OBSOLETE_GRAPHS,
        show_only: bool = False,
        scanned_params_order: Iterable[str] | None = None,
        show_titles: bool = True,
        orientation: str = "horizontal",
        highlight_pauli_compression: bool = False,
        show_stats_table: bool = False,
        save_tag: str | None = None,
        descriptive_legend: bool = True
    ) -> None:
        """Plot the Laplacian spectra for each configuration.

        Args:
            plot_window_start: Start as a fraction of spectrum length (0.0-1.0).
            plot_window_ends: End as a fraction of spectrum length (0.0-1.0).
            merge_plots: If True, create a merged grid of all configurations.
            exclude_graphs: Graph types to skip in plots.
            show_only: If True, show the plots instead of saving them to disk.
            highlight_pauli_compression: If True, annotate each plot with a box
                showing the Pauli-count gap between the Randomly Perturbed and
                Erdős-Rényi graphs and their compression ratio.
            show_stats_table: If True, legend entries show only graph names and a
                comparison table with |E|, |P|, |P_G| is rendered below the axes.
            save_tag: If provided, appended to output filenames as ``__<tag>``
                before the ``.png`` extension (e.g. ``spectra_plot__NO_TITLE.png``).
        """
        if not hasattr(self, "manifest_data") or not self.manifest_data:            
            items = []
            for i, res in enumerate(self.data):
                cfg_json = _config_to_jsonable(res["configuration"])
                item_id = f"{i:04d}-{_derive_item_slug(cfg_json, fallback=f'item-{i}')}"
                items.append({
                    "item_idx": i,
                    "item_id": item_id,
                    "config_index": res.get("config_index", i),
                    "configuration": cfg_json
                })
            self.manifest_data = {"items": items}

        # --- where to save ---
        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- plotting params ---
        default_markers = ['o', 's', '^', 'D', '+', 'x', 'P', '*']
        marker_cycler = cycle(default_markers)

        def _fmt_num(val, threshold: int = 999) -> str:
            if not isinstance(val, int) or val <= threshold:
                return str(val)
            exp = int(np.floor(np.log10(val)))
            coeff = val / 10 ** exp
            return f"${coeff:.3g}\\times10^{{{exp}}}$"

        # --- figure grid if merge_plots ---
        configs_list = [res["configuration"] for res in self.data] if self.data else list(self.configurations)
        num_configs = len(self.data)  # safer than len(configs_list) if partially saved/loaded
        
        used_axes = set()
        if merge_plots:
            _merge_base_figsize = (5, 5.5) if show_stats_table else (5, 4)
            merged_fig, axes_map, scanned_params, unique_values = self._create_merged_figure(
                configs_list, _merge_base_figsize, scanned_params_order, orientation=orientation,
                restrict_params=self._get_scanned_params_from_config()
            )

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
            _figsize = (8, 8) if show_stats_table else (6.4, 4.8)
            plt.figure(figsize=_figsize, dpi=FIGURES_DPI)

            ax = None
            if merge_plots:
                coord = tuple(unique_values[p].index(getattr(config, p)) for p in scanned_params)
                ax = axes_map[coord]
                used_axes.add(ax)

            graph_colors = {}
            stats_rows = []
            stats_graph_types = []

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
                label = graph_label_name
                if show_stats_table:
                    stats_rows.append((
                        graph_label_name,
                        bundle.metadata.num_edges,
                        bundle.num_laplacian_paulis,
                        bundle.num_commuting_groups,
                    ))
                    stats_graph_types.append(graph_type)
                if descriptive_legend:
                    label = (
                        f"{label} ($|E| = $ {_fmt_num(bundle.metadata.num_edges)}, "
                        f"$|P| = ${_fmt_num(bundle.num_laplacian_paulis)}, "
                        f"$|P_G| = ${_fmt_num(bundle.num_commuting_groups)})"
                    )

                sc = plt.scatter(
                    nodes_indexes[window_start:window_ends],
                    np.asarray(spec)[window_start:window_ends],
                    s=scatter_size,
                    label=label,
                    marker=marker,
                )
                fc = sc.get_facecolor()
                if len(fc):
                    graph_colors[graph_type] = tuple(fc[0])

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

            # Resolve Pauli counts + actual scatter colors for the compression box
            pauli_ann_data = None
            if highlight_pauli_compression:
                _rp_key = "random_order_scrambled_perturbed_graph"
                _er_key = "scrambled_like_random_graph_same_weights"
                rp_bundle = config_execution_result.get(_rp_key)
                er_bundle = config_execution_result.get(_er_key)
                if rp_bundle is not None and er_bundle is not None:
                    rp_p = rp_bundle.num_laplacian_paulis
                    er_p = er_bundle.num_laplacian_paulis
                    if isinstance(rp_p, int) and isinstance(er_p, int) and rp_p > 0:
                        pauli_ann_data = (
                            rp_p, er_p,
                            graph_colors.get(_rp_key, "#ff7f0e"),
                            graph_colors.get(_er_key, "#2ca02c"),
                        )

            def _attach_stats_table(target_ax, rows, row_colors=None, fontsize: int = 8) -> None:
                col_labels = ["Graph", "#Edges $|E|$", "#Pauli strings $|P|$", "#Commuting groups $|P_G|$"]
                cell_text = [
                    [name, _fmt_num(e), _fmt_num(p), _fmt_num(pg)]
                    for name, e, p, pg in rows
                ]
                tbl = target_ax.table(
                    cellText=cell_text,
                    colLabels=col_labels,
                    cellLoc="center",
                    loc="bottom",
                    bbox=[0, -0.66, 1, 0.44],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(fontsize)

                def _lighten(rgba):
                    r, g, b = rgba[:3]
                    return (r * 0.22 + 0.78, g * 0.22 + 0.78, b * 0.22 + 0.78, 1.0)

                for (row, col), cell in tbl.get_celld().items():
                    cell.set_edgecolor("#cccccc")
                    if row == 0:
                        cell.set_facecolor("#e8e8e8")
                        cell.set_text_props(fontweight="bold")
                    elif row_colors is not None and (row - 1) < len(row_colors):
                        color = row_colors[row - 1]
                        if color is not None:
                            # cell.set_facecolor(_lighten(color))
                            if col == 0:
                                cell.set_text_props(color=color[:3], fontweight="bold")

                fig = target_ax.get_figure()
                try:
                    from matplotlib.layout_engine import ConstrainedLayoutEngine
                    _is_constrained = isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
                except (ImportError, AttributeError):
                    _is_constrained = getattr(fig, "_constrained", False)
                if not _is_constrained:
                    fig.subplots_adjust(bottom=0.44)

            def _attach_pauli_box(target_ax, rp_p, er_p, rp_c, er_c, fontsize: int = 9):
                ratio = er_p / rp_p
                areas = VPacker(align="left", pad=4, sep=5, children=[
                    TextArea("#Pauli terms", textprops=dict(
                        fontsize=fontsize, fontweight="bold", color="#333333"
                    )),
                    TextArea(f"ROP: {rp_p:,}", textprops=dict(
                        fontsize=fontsize, color=rp_c, fontweight="bold"
                    )),
                    TextArea(f"ER: {er_p:,}", textprops=dict(
                        fontsize=fontsize, color=er_c, fontweight="bold"
                    )),
                    TextArea(f"×{ratio:,.0f}", textprops=dict(
                        fontsize=max(fontsize - 1, 7), color="#555555", style="italic"
                    )),
                ])
                ab = AnchoredOffsetbox(
                    loc="upper left",
                    bbox_to_anchor=(0.65, 0.47),
                    bbox_transform=target_ax.transAxes,
                    child=areas,
                    frameon=True,
                )
                ab.patch.set_edgecolor("#aaaaaa")
                ab.patch.set_facecolor("#f8f8f8")
                ab.patch.set_alpha(0.93)
                target_ax.add_artist(ab)

            plt.xlabel("Eigenvalue index")
            plt.ylabel("Eigenvalue")
            plt.legend(
                loc="upper left",
                frameon=True,
                fontsize=9,
                markerscale=3,
                framealpha=0.9
            )
            plt.grid(True, linestyle="--", alpha=0.4)

            if show_titles:
                plt.title(title_str, fontsize=8)

            if pauli_ann_data is not None:
                _attach_pauli_box(plt.gca(), *pauli_ann_data)

            if show_stats_table and stats_rows:
                _row_colors = [graph_colors.get(gt) for gt in stats_graph_types]
                _attach_stats_table(plt.gca(), stats_rows, row_colors=_row_colors)

            if not show_only:
                stem = f"spectra_plot__{save_tag}" if save_tag else "spectra_plot"
                out_png = Path(run_dir, manifest_item["item_id"], f"{stem}.png")
                out_png.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_png, dpi=FIGURES_DPI, bbox_inches="tight")
            plt.close()

            # format merged figure
            if ax is not None:
                ax.set_xlabel("Eigenvalue index")
                ax.set_ylabel("Eigenvalue")
                ax.grid(True, linestyle="--", alpha=0.4)

                if show_titles:
                    ax.set_title(title_str, fontsize=8)

                ax.legend(
                    loc="upper left",
                    frameon=True,
                    fontsize=7,
                    markerscale=2,
                    framealpha=0.9
                )
                if pauli_ann_data is not None:
                    _attach_pauli_box(ax, *pauli_ann_data)

                if show_stats_table and stats_rows:
                    _row_colors = [graph_colors.get(gt) for gt in stats_graph_types]
                    _attach_stats_table(ax, stats_rows, row_colors=_row_colors, fontsize=7)

        if merge_plots:
            # Hide any unused axes in sparse parameter grids
            for coord, ax_item in axes_map.items():
                if ax_item not in used_axes:
                    ax_item.set_axis_off() # Removes the axes lines while preserving grid alignment
            
            # REMOVE the merged_fig.tight_layout() block. # TODO CHECK THIS
            # It causes layout engine conflicts when used with 'constrained'.
            
            if not show_only:
                merged_stem = f"merged_spectra_plot__{save_tag}" if save_tag else "merged_spectra_plot"
                merged_path = Path(run_dir, f"{merged_stem}.png")
                merged_fig.savefig(merged_path, dpi=FIGURES_DPI, bbox_inches="tight")
                plt.close(merged_fig)

        if show_only:
            plt.show()

    def plot_matrices(
        self,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = OBSOLETE_GRAPHS,
        show_only: bool = False,
        scanned_params_order: Iterable[str] | None = None,
        orientation: str = "horizontal",
        show_titles: bool = True,
        save_tag: str | None = None,
    ) -> None:
        """Create sparsity pattern visualizations of Laplacian matrices.
        
        Args:
            merge_plots: If True, create a merged grid of all configurations per graph type.
            exclude_graphs: Graph types to skip in visualization.
            show_only: If True, show the plots instead of saving them to disk.
        """
        if not hasattr(self, "manifest_data") or not self.manifest_data:
            items = []
            for i, res in enumerate(self.data):
                cfg_json = _config_to_jsonable(res["configuration"])
                item_id = f"{i:04d}-{_derive_item_slug(cfg_json, fallback=f'item-{i}')}"
                items.append({
                    "item_idx": i,
                    "item_id": item_id,
                    "config_index": res.get("config_index", i),
                    "configuration": cfg_json
                })
            self.manifest_data = {"items": items}

        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        configs_list = [res["configuration"] for res in self.data] if self.data else list(self.configurations)
        
        merged_figs = {}
        merged_axes_maps = {}
        scanned_params = []
        unique_values = {}
        used_axes = {g: set() for g in GRAPH_TYPES}
        
        if merge_plots:
            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs:
                    continue
                fig, axes_map, sp, uv = self._create_merged_figure(
                    configs_list, (6, 4), scanned_params_order, orientation=orientation,
                    restrict_params=self._get_scanned_params_from_config()
                )
                merged_figs[graph_name] = fig
                merged_axes_maps[graph_name] = axes_map
                scanned_params = sp
                unique_values = uv

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
                plt.figure(dpi=FIGURES_DPI)
                plt.spy(config_data[graph_name].laplacian_dense_matrix, markersize=0.1)
                if show_titles:
                    plt.title(f"{graph_name} - {config}")
                if not show_only:
                    filename = f"{graph_name}_laplacian"
                    if save_tag is not None:
                        filename = f"{filename}_{save_tag}"
                    filename = f"{filename}.png"

                    plt.savefig(Path(config_data_path, filename), dpi=FIGURES_DPI)
                    plt.close()

                # Merged plot
                if merge_plots and graph_name in merged_axes_maps:
                    coord = tuple(unique_values[p].index(getattr(config, p)) for p in scanned_params)
                    ax = merged_axes_maps[graph_name][coord]
                    used_axes[graph_name].add(ax)
                    
                    L = config_data[graph_name].laplacian_dense_matrix
                    ax.spy(L, markersize=0.2)
                    if show_titles:
                        ax.set_title(f"Config: {config}, nonzero_rate = {np.count_nonzero(L) / L.size:.2f}", fontsize=8)

        if merge_plots:
            for graph_name, fig in merged_figs.items():
                axes_map = merged_axes_maps[graph_name]
                for coord, ax_item in axes_map.items():
                    if ax_item not in used_axes[graph_name]:
                        ax_item.set_visible(False)
                
                try:
                    fig.tight_layout()
                except Exception:
                    pass

                if not show_only:
                    filename = f"merged_{graph_name}_matrices"
                    if save_tag is not None:
                        filename = f"{filename}_{save_tag}"
                    filename = f"{filename}.png"

                    merged_path = Path(run_dir, filename)
                    fig.savefig(merged_path, dpi=FIGURES_DPI, bbox_inches="tight")
                    plt.close(fig)

        if show_only:
            plt.show()

    def draw_graphs(
        self,
        merge_plots: bool = True,
        exclude_graphs: Iterable[str] = OBSOLETE_GRAPHS,
        show_only: bool = False,
        scanned_params_order: Iterable[str] | None = None,
        orientation: str = "horizontal",
        show_titles: bool = True,
        save_tag: str | None = None,
    ) -> None:
        """Draw the graph structures using NetworkX.
        
        Args:
            merge_plots: If True, create a merged grid of all configurations per graph type.
            exclude_graphs: Graph types to skip in visualization.
            show_only: If True, show the plots instead of saving them to disk.
        """
        if not hasattr(self, "manifest_data") or not self.manifest_data:
            items = []
            for i, res in enumerate(self.data):
                cfg_json = _config_to_jsonable(res["configuration"])
                item_id = f"{i:04d}-{_derive_item_slug(cfg_json, fallback=f'item-{i}')}"
                items.append({
                    "item_idx": i,
                    "item_id": item_id,
                    "config_index": res.get("config_index", i),
                    "configuration": cfg_json
                })
            self.manifest_data = {"items": items}

        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)

        configs_list = [res["configuration"] for res in self.data] if self.data else list(self.configurations)

        merged_figs = {}
        merged_axes_maps = {}
        merged_figs_last_nodes = {}
        scanned_params = []
        unique_values = {}
        used_axes = {g: set() for g in GRAPH_TYPES}

        if merge_plots:
            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs:
                    continue
                fig, axes_map, sp, uv = self._create_merged_figure(
                    configs_list, (7, 5), scanned_params_order, orientation=orientation,
                    restrict_params=self._get_scanned_params_from_config()
                )
                merged_figs[graph_name] = fig
                merged_axes_maps[graph_name] = axes_map
                scanned_params = sp
                unique_values = uv

        # Pre-compute global degree range per graph type for a consistent colorbar scale
        # across all subplots in the merged figure.
        global_degree_min: dict[str, float] = {}
        global_degree_max: dict[str, float] = {}
        for config_data in self.data:
            for graph_name in GRAPH_TYPES:
                if graph_name in exclude_graphs or graph_name not in config_data:
                    continue
                graph_obj = config_data[graph_name].graph_obj
                if graph_obj is None:
                    continue
                degs = [d for _, d in graph_obj.degree(weight="weight")]
                if degs:
                    global_degree_min[graph_name] = min(global_degree_min.get(graph_name, float("inf")), min(degs))
                    global_degree_max[graph_name] = max(global_degree_max.get(graph_name, float("-inf")), max(degs))

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
                plt.figure(figsize=(8, 6), dpi=FIGURES_DPI)
                ax = plt.gca()
                nodes = nx.draw_networkx_nodes(
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
                plt.colorbar(nodes, ax=ax, label="Weighted degree", shrink=0.7)
                ax.axis("off")
                if show_titles:
                    plt.title(f"{graph_name} - {config}")

                if not show_only:
                    filename = graph_name
                    if save_tag is not None:
                        filename = f"{filename}_{save_tag}"
                    filename = f"{filename}_graph_draw.png"

                    plt.savefig(Path(config_data_path, filename), dpi=FIGURES_DPI, bbox_inches="tight")
                    plt.close()

                # Merged plot
                if merge_plots and graph_name in merged_axes_maps:
                    coord = tuple(unique_values[p].index(getattr(config, p)) for p in scanned_params)
                    ax = merged_axes_maps[graph_name][coord]
                    used_axes[graph_name].add(ax)

                    merged_nodes = nx.draw_networkx_nodes(
                        graph_obj, pos, ax=ax,
                        node_size=15,
                        node_color=degrees,
                        cmap=plt.cm.plasma,
                        vmin=global_degree_min.get(graph_name),
                        vmax=global_degree_max.get(graph_name),
                        alpha=0.9,
                        edgecolors="white",
                        linewidths=0.3
                    )
                    merged_figs_last_nodes[graph_name] = merged_nodes
                    nx.draw_networkx_edges(
                        graph_obj, pos, ax=ax,
                        alpha=edge_alpha,
                        edge_color="gray",
                        width=0.5
                    )
                    ax.axis("off")
                    if show_titles:
                        ax.set_title(f"Config: {config}", fontsize=9)

        if merge_plots:
            for graph_name, fig in merged_figs.items():
                axes_map = merged_axes_maps[graph_name]
                for coord, ax_item in axes_map.items():
                    if ax_item not in used_axes[graph_name]:
                        ax_item.set_visible(False)

                if graph_name in merged_figs_last_nodes:
                    fig.colorbar(merged_figs_last_nodes[graph_name], ax=list(axes_map.values()), label="Weighted degree", shrink=0.6)

                try:
                    fig.tight_layout()
                except Exception:
                    pass
                if not show_only:
                    filename = f"merged_{graph_name}"
                    if save_tag is not None:
                        filename = f"{filename}_{save_tag}"
                    filename = f"{filename}_graphs.png"

                    merged_path = Path(run_dir, filename)
                    fig.savefig(merged_path, dpi=FIGURES_DPI, bbox_inches="tight")
                    plt.close(fig)

        if show_only:
            plt.show()

    def plot_soergel_distance(
        self,
        show_only: bool = False,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        """Plot the Soergel distance as a function of max_perturbation_locality (m)
        for every configuration of (n_num_qubits q, num_perturbations n_p).

        Args:
            show_only: If True, display the plot using plt.show() and do not save it.
            figsize: The dimensions of the resulting figure.
        """
        # Ensure we have data and it has been analyzed
        if not self.data:
            logger.warning("No data available to plot Soergel distance. Run perform_experiment() first.")
            return

        points = []
        for config_data in self.data:
            cfg = config_data.get("configuration")
            if not cfg:
                continue
            comparison = config_data.get("spectra_comparison")
            # If not yet analyzed, try running analyze_results on the fly
            if comparison is None:
                self.analyze_results()
                comparison = config_data.get("spectra_comparison")
            
            if comparison is None or "soergel_distance" not in comparison:
                continue
                
            soergel = comparison["soergel_distance"]
            
            # JSON deserialization turns tuples into lists, which are unhashable
            weights_val = cfg.perturbation_weights_bounds
            if isinstance(weights_val, list):
                weights_val = tuple(weights_val)
                
            points.append({
                "q": cfg.n_num_qubits,
                "n_p": cfg.num_perturbations,
                "m": cfg.max_perturbation_locality,
                "soergel": soergel,
                "d": cfg.d_skeleton_regularity,
                "sl": cfg.max_skeleton_locality,
                "weights": weights_val,
                "seed": cfg.seed
            })

        if not points:
            logger.warning("No Soergel distance data found in experiment results.")
            return

        # Identify which fields actually vary to keep the legend neat
        fields = ["q", "n_p", "d", "sl", "weights", "seed", "m"]
        unique_vals = {f: set() for f in fields}
        for p in points:
            for f in fields:
                unique_vals[f].add(p[f])

        # Group points by the full parameter set (excluding 'm' and 'soergel')
        groups = {}
        for p in points:
            key = (p["q"], p["n_p"], p["d"], p["sl"], p["weights"], p["seed"])
            if key not in groups:
                groups[key] = []
            groups[key].append((p["m"], p["soergel"]))

        # Sort each group by 'm' (max_perturbation_locality)
        for key in groups:
            groups[key].sort(key=lambda x: x[0])

        # Setup premium plotting style
        plt.figure(figsize=figsize, dpi=FIGURES_DPI)
        
        # Color palette: dynamic based on number of groups using a modern colormap
        num_curves = len(groups)
        cmap = plt.get_cmap("plasma")
        
        # Access style-friendly markers
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "P", "*", "X", "h"]
        
        for idx, (key, xy_points) in enumerate(sorted(groups.items())):
            q_val, np_val, d_val, sl_val, w_val, seed_val = key
            
            # Construct label
            label_parts = [f"$q={q_val}$", f"$n_p={np_val}$"]
            if len(unique_vals["d"]) > 1:
                label_parts.append(f"$d={d_val}$")
            if len(unique_vals["sl"]) > 1:
                label_parts.append(f"$sl={sl_val}$")
            if len(unique_vals["weights"]) > 1:
                label_parts.append(f"$w={w_val}$")
            if len(unique_vals["seed"]) > 1:
                label_parts.append(f"$seed={seed_val}$")
                
            label = ", ".join(label_parts)
            
            x_vals = [pt[0] for pt in xy_points]
            y_vals = [pt[1] for pt in xy_points]
            
            color = cmap(idx / (num_curves - 1)) if num_curves > 1 else cmap(0)
            marker = markers[idx % len(markers)]
            
            plt.plot(
                x_vals, 
                y_vals, 
                label=label, 
                marker=marker, 
                color=color, 
                linewidth=2.0, 
                markersize=8, 
                markeredgecolor="white", 
                markeredgewidth=1.0,
                alpha=0.85
            )

        # Plot labels and formatting
        plt.xlabel("Maximum Perturbation Locality ($m$)", fontsize=12, fontweight="bold", labelpad=10)
        plt.ylabel("Soergel Distance (Normalized $L_1$)", fontsize=12, fontweight="bold", labelpad=10)
        
        # Clean, modern title
        plt.title(
            "Eigenspectrum Similarity (Soergel Distance) vs. Perturbation Locality ($m$)\n"
            "Comparing Randomly Perturbed Laplacian to Classical ER Graph", 
            fontsize=13, 
            fontweight="bold", 
            pad=15
        )
        
        # Elegant grid and ticks
        plt.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
        plt.gca().tick_params(axis='both', which='major', labelsize=10)
        
        # Set integer ticks on the X-axis since locality 'm' is an integer
        all_m_values = sorted(list(unique_vals["m"]))
        plt.xticks(all_m_values)
        
        # Move legend cleanly to the side or upper right with glassmorphism-like feel
        plt.legend(
            loc="center left", 
            bbox_to_anchor=(1.02, 0.5), 
            title="Configurations",
            title_fontsize=11,
            frameon=True, 
            facecolor="#f9f9f9", 
            edgecolor="#dddddd",
            framealpha=0.95,
            fontsize=10
        )
        
        plt.tight_layout()

        # Handle saving / showing
        run_dir = Path(self.metadata.get("run_metadata", {}).get("run_dir", "."))
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if not show_only:
            out_png = run_dir / "soergel_distance_vs_locality.png"
            plt.savefig(out_png, dpi=FIGURES_DPI, bbox_inches="tight")
            logger.info(f"Saved Soergel distance plot to: {out_png}")
            plt.close()
        else:
            plt.show()

    def run_all(
        self,
        filepath: str,
        draw_graphs: bool = False,
        figures_orientation: str = "horizontal",
        show_titles: bool = True,
        highlight_pauli_compression: bool = False,
    ) -> None:
        """Execute the complete experiment pipeline.

        Args:
            filepath: Path where results will be saved.
            figures_orientation: Orientation of the figures (horizontal/vertical).
            highlight_pauli_compression: Passed through to plot_results.
        """
        try:
            self.perform_experiment(save_dir=filepath)
        except Exception as e:
            if self.data:
                logger.info(f"Experiment pipeline encountered an error, but attempting to generate plots for the {len(self.data)} successfully completed configurations.")
                try:
                    self.plot_results(
                        orientation=figures_orientation,
                        show_titles=show_titles,
                        highlight_pauli_compression=highlight_pauli_compression
                    )
                    self.plot_matrices(orientation=figures_orientation)
                    self.plot_soergel_distance()
                    if draw_graphs:
                        self.draw_graphs(orientation=figures_orientation)
                except Exception as plot_err:
                    logger.error(f"Failed to generate plots for the partial run: {plot_err}")
            raise e

        self.plot_results(orientation=figures_orientation, highlight_pauli_compression=highlight_pauli_compression)
        self.plot_matrices(orientation=figures_orientation)
        self.plot_soergel_distance()
        if draw_graphs:
            self.draw_graphs(orientation=figures_orientation)


if __name__ == "__main__":

    # for q in [10]:
    #     ec = ExperimentConfigurations(
    #         n_num_qubits=[q], # q
    #         d_skeleton_regularity=[3],
    #         max_skeleton_locality=[3],
    #         num_perturbations=[
    #             0,
    #             lambda x: int(np.sqrt(x)),
    #             lambda x: x,
    #             lambda x: x**2,
    #             lambda x: 2**x,
    #             # lambda x: 2 * 2**x
    #         ],
    #         max_perturbation_locality=[3, 4], #list(range(1, q + 1)), # m
    #         perturbation_weights_bounds=[(0.5, 5)],
    #         seed=[32],
    #     )

    #     experiment = LaplacianHamiltoniansWorkshop(configurations=ec)
    #     experiment.run_all("experiments_data_archive", draw_graphs=True)

    # ec = ExperimentConfigurations(
    #     n_num_qubits=[7, 9, 11, 13], # q
    #     d_skeleton_regularity=[3],
    #     max_skeleton_locality=[3],
    #     num_perturbations=[
    #         # 0,
    #         # lambda x: int(np.sqrt(x)),
    #         lambda x: x,
    #         lambda x: x**2,
    #         lambda x: 2**x,
    #     ],
    #     max_perturbation_locality=[3, 4, 5, 6], # m
    #     perturbation_weights_bounds=[(0.5, 5)],
    #     seed=[32],
    # )

    # ec = ExperimentConfigurations(
    #     n_num_qubits=[12], # q
    #     d_skeleton_regularity=[3],
    #     max_skeleton_locality=[3],
    #     num_perturbations=[
    #         # 0,
    #         # lambda x: int(np.sqrt(x)),
    #         # lambda x: x,
    #         lambda x: x**2,
    #         # lambda x: 2**x,
    #     ],
    #     max_perturbation_locality=[4], # m
    #     perturbation_weights_bounds=[(0.5, 5)],
    #     seed=[32],
    # )

    # experiment = LaplacianHamiltoniansWorkshop(configurations=ec)
    # experiment.run_all("experiments_data_archive", draw_graphs=False)

    data_dir_path = Path(
        "/home/ohad-lev/ohad/msc/research/thesis/qsga/experiments_data_archive"
    )
    save_tag = "FOR_PAPER"

    experiment = LaplacianHamiltoniansWorkshop.from_data(Path(data_dir_path, "2026-05-27_09-35-22"))
    experiment.analyze_results()
    experiment.plot_results()#show_stats_table=True, save_tag="WITH_STATS_TABLE")#show_titles=False, highlight_pauli_compression=False, save_tag=save_tag)
    # experiment.plot_matrices()#show_titles=False, save_tag=save_tag)
    # experiment.plot_soergel_distance()
    # experiment.draw_graphs()#show_titles=False, save_tag=save_tag)