from dataclasses import dataclass
from pathlib import Path

import numpy as np
import networkx as nx

from qsga.data_handling import (
    _jsonify_complex,
    _jsonify_seq_with_complex,
    _maybe_asdict,
    _collect_bundle_attrs_for_manifest,
    _bundle_get,
    _now_iso,
    _config_to_jsonable,
    _slugify,
    _derive_item_slug,
    _save_graph,
    _load_graph,
    _save_array,
    _load_array,
    _save_sparse_matrix,
    _load_sparse_matrix_to_dense,
    save_dataset,
    load_dataset,
    load_experiment_metadata
)


VERBOSE = True


@dataclass
class DummyDataClass:
    a: int
    b: str


def test_jsonify_complex():
    # Python complex
    c1 = 1 + 2j
    assert _jsonify_complex(c1) == {"re": 1.0, "im": 2.0}
    
    # NumPy complex
    c2 = np.complex128(3 - 4j)
    assert _jsonify_complex(c2) == {"re": 3.0, "im": -4.0}
    
    # Regular value
    assert _jsonify_complex(5) == 5

    if VERBOSE:
        print(f"\n--- test_jsonify_complex ---")


def test_jsonify_seq_with_complex():
    seq = [1, 2+3j, [4, 5-6j]]
    res = _jsonify_seq_with_complex(seq)
    assert res == [1, {"re": 2.0, "im": 3.0}, [4, {"re": 5.0, "im": -6.0}]]

    if VERBOSE:
        print(f"\n--- test_jsonify_seq_with_complex ---")


def test_maybe_asdict():
    dc = DummyDataClass(a=1, b="test")
    assert _maybe_asdict(dc) == {"a": 1, "b": "test"}
    assert _maybe_asdict({"x": 1}) == {"x": 1}

    if VERBOSE:
        print(f"\n--- test_maybe_asdict ---")


def test_collect_bundle_attrs_for_manifest():
    class DummyBundle:
        laplacian_pauli_repr = [("IX", 1+0j)]
        laplacian_sparse_pauli_repr = [("IX", [0, 1], 1+0j)]
        metadata = DummyDataClass(1, "test")
        seed = 42
        laplacian_dense_matrix = np.array([1])
    
    attrs = _collect_bundle_attrs_for_manifest(DummyBundle())
    assert attrs["seed"] == 42
    assert "laplacian_dense_matrix" not in attrs
    assert attrs["laplacian_pauli_repr"] == [["IX", {"re": 1.0, "im": 0.0}]]

    if VERBOSE:
        print(f"\n--- test_collect_bundle_attrs_for_manifest ---")


def test_bundle_get():
    class Dummy:
        a = 1
    assert _bundle_get(Dummy(), "a") == 1
    assert _bundle_get({"a": 2}, "a") == 2
    assert _bundle_get({}, "b", 3) == 3

    if VERBOSE:
        print(f"\n--- test_bundle_get ---")


def test_now_iso():
    iso = _now_iso()
    assert isinstance(iso, str)
    assert "T" in iso

    if VERBOSE:
        print(f"\n--- test_now_iso ---")


def test_config_to_jsonable():
    assert _config_to_jsonable(DummyDataClass(1, "t")) == {"a": 1, "b": "t"}
    assert _config_to_jsonable({"x": 1}) == {"x": 1}

    if VERBOSE:
        print(f"\n--- test_config_to_jsonable ---")


def test_slugify():
    assert _slugify("Hello World!") == "hello-world"
    assert _slugify("A" * 60, max_len=10) == "a" * 10
    assert _slugify("___") == "item"

    if VERBOSE:
        print(f"\n--- test_slugify ---")


def test_derive_item_slug():
    cfg = {
        "n_num_qubits": 2,
        "d_skeleton_regularity": 3,
        "max_skeleton_locality": 4,
        "num_perturbations": 5,
        "max_perturbation_locality": 6,
        "seed": 42
    }
    slug1 = _derive_item_slug(cfg, "fallback")
    assert slug1 == "q2-d3-sl4-np5-m6-s42"
    assert _derive_item_slug(None, "fallback") == "fallback"

    if VERBOSE:
        print(f"\n--- test_derive_item_slug ---")
        print(f"Config:\n{cfg}\nDerived Slug: {slug1}")


def test_save_load_graph(tmp_path: Path):
    g = nx.Graph()
    g.add_edge(1, 2, weight=0.5)
    
    path = tmp_path / "graph.json"
    _save_graph(g, path)
    
    g2 = _load_graph(path)
    assert g2.number_of_nodes() == 2
    assert g2.number_of_edges() == 1
    assert g2[1][2]["weight"] == 0.5

    if VERBOSE:
        print(f"\n--- test_save_load_graph ---")
        print(f"Original edges: {list(g.edges(data=True))}")
        print(f"Loaded edges: {list(g2.edges(data=True))}")


def test_save_load_array(tmp_path: Path):
    arr = np.array([1, 2, 3])
    path = tmp_path / "arr.npy"
    _save_array(arr, path)
    
    arr2 = _load_array(path)
    assert np.array_equal(arr, arr2)

    if VERBOSE:
        print(f"\n--- test_save_load_array ---")
        print(f"Array: {arr}")


def test_save_load_sparse_matrix(tmp_path: Path):
    arr = np.array([[1, 0], [0, 2]])
    path = tmp_path / "arr.npz"
    _save_sparse_matrix(arr, path)
    
    arr2 = _load_sparse_matrix_to_dense(path)
    assert np.array_equal(arr, arr2)

    if VERBOSE:
        print(f"\n--- test_save_load_sparse_matrix ---")
        print(f"Original Matrix:\n{arr}")
        print(f"Loaded Dense Matrix:\n{arr2}")


def test_save_load_dataset(tmp_path: Path):
    class DummyGraphData:
        def __init__(self):
            self.graph_obj = nx.Graph()
            self.graph_obj.add_edge(1, 2)
            self.laplacian_dense_matrix = np.array([[1, -1], [-1, 1]])
            self.laplacian_spectrum = [0.0, 2.0]
            self.metadata = DummyDataClass(1, "test")
            self.laplacian_pauli_repr = [("IX", 1+0j)]

    # We mock out GRAPH_TYPES dynamically
    import qsga.data_handling
    original_types = qsga.data_handling.GRAPH_TYPES
    qsga.data_handling.GRAPH_TYPES = ["skeleton_graph"]

    data = [{
        "config_index": 0,
        "configuration": {"n_num_qubits": 2},
        "skeleton_graph": DummyGraphData()
    }]
    
    manifest = save_dataset(
        data=data,
        out_dir=tmp_path,
        run_name="test_run",
        experiment_metadata={"test": "meta"}
    )
    
    assert "items" in manifest
    assert len(manifest["items"]) == 1
    
    run_dir = list(tmp_path.iterdir())[0] # The timestamp dir
    
    rebuilt_data, rebuilt_manifest, rebuilt_meta = load_dataset(run_dir)
    assert len(rebuilt_data) == 1
    assert rebuilt_meta["test"] == "meta"
    
    b = rebuilt_data[0]["skeleton_graph"]
    assert isinstance(b["graph_obj"], nx.Graph)
    assert np.array_equal(b["laplacian_obj"], [[1, -1], [-1, 1]])
    assert np.array_equal(b["laplacian_spectrum"], [0.0, 2.0])
    
    loaded_meta = load_experiment_metadata(run_dir)
    assert loaded_meta["test"] == "meta"
    
    # Restore GRAPH_TYPES
    qsga.data_handling.GRAPH_TYPES = original_types

    if VERBOSE:
        print(f"\n--- test_save_load_dataset ---")
        print(f"Created manifest with {len(manifest['items'])} items")
        print(f"Loaded Meta: {loaded_meta}")
