"""Vibe-coded with ChatGPT."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Callable

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


GRAPH_TYPES = [
    "skeleton_graph",
    "definite_order_perturbated_graph", 
    "random_order_perturbated_graph",
    "dop_like_random_graph", # Unweighted and weighted densities should be as `random_order_perturbated_graph`
    "rop_like_random_graph", # Unweighted and weighted densities should be as `random_order_perturbated_graph`
]


# ----------------------- utilities -----------------------


def _jsonify_complex(z: Any) -> Any:
    # Convert complex (Python or NumPy) into {"re": float, "im": float}
    if isinstance(z, complex):
        return {"re": float(z.real), "im": float(z.imag)}
    # numpy complex
    if hasattr(z, "real") and hasattr(z, "imag") and hasattr(z, "dtype") and "complex" in str(z.dtype):
        return {"re": float(np.real(z)), "im": float(np.imag(z))}
    return z


def _jsonify_seq_with_complex(seq: Iterable[Any]) -> list[Any]:
    out = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            out.append(_jsonify_seq_with_complex(item))
        else:
            out.append(_jsonify_complex(item))
    return out


def _maybe_asdict(x: Any) -> Any:
    try:
        return asdict(x) if is_dataclass(x) else x
    except Exception:
        return x


def _collect_bundle_attrs_for_manifest(bundle: Any) -> dict[str, Any]:
    """
    Collect all GraphData-like attributes to embed in manifest.json,
    excluding the heavy fields:
      - laplacian_dense_matrix
      - laplacian_spectrum
    """
    # Access attrs from object or dict
    def g(name, default=None):
        if isinstance(bundle, dict):
            return bundle.get(name, default)
        return getattr(bundle, name, default)

    attrs: dict[str, Any] = {}

    # Laplacian Pauli representations (convert complex numbers to JSONable)
    lpr = g("laplacian_pauli_repr")
    if lpr is not None:
        # expected: list[tuple[str, complex]]
        attrs["laplacian_pauli_repr"] = _jsonify_seq_with_complex(lpr)

    lspr = g("laplacian_sparse_pauli_repr")
    if lspr is not None:
        # expected: list[tuple[str, list[int], complex]]
        attrs["laplacian_sparse_pauli_repr"] = _jsonify_seq_with_complex(lspr)

    # Metadata (dataclass → dict)
    meta = g("metadata")
    if meta is not None:
        attrs["metadata"] = _maybe_asdict(meta)

    # You might want to persist seed-like info if present (optional)
    for opt in ("seed",):
        v = g(opt)
        if v is not None:
            attrs[opt] = v

    # DO NOT include laplacian_dense_matrix or laplacian_spectrum here
    # They are saved as files (npy, csv)

    return attrs


def _bundle_get(bundle: Any, name: str, default: Any = None) -> Any:
    # Try attribute
    if hasattr(bundle, name):
        return getattr(bundle, name)
    # Try mapping
    if isinstance(bundle, dict) and name in bundle:
        return bundle[name]
    return default


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _config_to_jsonable(cfg: Any) -> Any:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "model_dump"):   # Pydantic v2
        return cfg.model_dump()
    if hasattr(cfg, "dict"):         # Pydantic v1
        return cfg.dict()
    try:
        json.dumps(cfg)
        return cfg
    except TypeError:
        return {"__repr__": repr(cfg)}


def _slugify(s: str, max_len: int = 48) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s or "item")[:max_len]


def _derive_item_slug(cfg_json: Any, fallback: str) -> str:
    """
    Create a short, human-readable, filesystem-safe slug for the configuration.
    Example → '0000-n8-d3-sl2-p4-pl3-s1'
    """
    try:
        if isinstance(cfg_json, dict):
            parts = []
            mapping = {
                "n_num_qubits": "n",
                "d_skeleton_regularity": "d",
                "max_skeleton_locality": "sl",
                "num_perturbations": "p",
                "max_perturbation_locality": "pl",
                "seed": "s",
            }
            for key, prefix in mapping.items():
                val = cfg_json.get(key)
                if val is not None:
                    parts.append(f"{prefix}{val if isinstance(val, (int, float, str)) else 'x'}")
            if parts:
                return _slugify("-".join(parts))
        return _slugify(str(cfg_json)[:64]) if cfg_json is not None else _slugify(fallback)
    except Exception:
        return _slugify(fallback)


def _save_graph(g: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(g)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False))


def _load_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text())
    return json_graph.node_link_graph(data)


def _save_array(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


# ----------------------- main API -----------------------


def save_dataset(
    data: Iterable[dict[str, Any]],
    out_dir: str | Path,
    *,
    run_name: str | None = None,
    run_notes: str | None = None,
    item_namer: Callable[[int, dict[str, Any]], str] | None = None,
    experiment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save dataset and metadata to a timestamped run directory.

    The experiment metadata (e.g., ExperimentConfigurations) is always written
    to `experiment_metadata.json` under a key `"configurations"`.
    """
    base_out_dir = Path(out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_suffix = f"_{_slugify(run_name)}" if run_name else ""
    run_dir = base_out_dir / f"{timestamp}{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "created_at": _now_iso(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        "run_name": run_name,
        "run_notes": run_notes,
        "run_dir": str(run_dir),
    }
    experiment_metadata["run_metadata"] = run_metadata

    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(experiment_metadata, indent=4, ensure_ascii=False)
    )
    # ----------------------------------

    manifest_items: list[dict[str, Any]] = []

    data = list(data)
    for idx, rec in enumerate(data):
        cfg_json = _config_to_jsonable(rec.get("configuration"))
        item_id = (item_namer(idx, rec)
                   if item_namer is not None
                   else f"{idx:04d}-{_derive_item_slug(cfg_json, fallback=f'item-{idx}')}")
        item_dir = run_dir / item_id
        item_dir.mkdir(exist_ok=True)

        # Save per-item metadata
        item_meta = {
            "item_idx": idx,
            "item_id": item_id,
            "created_at": _now_iso(),
            "config_index": rec.get("config_index", idx),
            # "num_nodes": int(rec.get("num_nodes")),
            "configuration": cfg_json,
        }
        (item_dir / "metadata.json").write_text(json.dumps(item_meta, indent=4, ensure_ascii=False))

        # Graph bundles and spectra
        spectra_by_bundle: dict[str, np.ndarray | None] = {}

        def handle_bundle(name: str):
            bundle = rec[name]
            entry = {}

            # --- graph_obj → pretty JSON file ---
            g = bundle.get("graph_obj") if isinstance(bundle, dict) else getattr(bundle, "graph_obj", None)
            if g is None:
                raise ValueError(f"{name}: graph_obj is missing; ensure GraphData.graph_obj is set.")
            g_path = item_dir / f"{name}.graph.json"
            _save_graph(g, g_path)
            entry["graph_json"] = str(g_path.relative_to(run_dir))

            # --- laplacian_obj (dense) → .npy on disk ---
            L = None
            if isinstance(bundle, dict):
                L = bundle.get("laplacian_dense_matrix")
            else:
                L = getattr(bundle, "laplacian_dense_matrix")

            L_path = item_dir / f"{name}.laplacian.npy"
            _save_array(L, L_path)
            entry["laplacian_npy"] = str(L_path.relative_to(run_dir))

            # --- spectrum (optional) → kept out of manifest; written to spectra.csv below ---
            spec = bundle.get("laplacian_spectrum") if isinstance(bundle, dict) else getattr(bundle, "laplacian_spectrum", None)
            if spec is None:
                entry["has_spectrum"] = False
                spectra_by_bundle[name] = None
            else:
                s = np.sort(np.asarray(spec).ravel())
                spectra_by_bundle[name] = s
                entry["has_spectrum"] = True

            # --- add lightweight GraphData attrs into manifest ---
            entry.update(_collect_bundle_attrs_for_manifest(bundle))

            return entry

        item_entry = {
            "item_idx": idx,
            "item_id": item_id,
            "config_index": rec.get("config_index", idx),
            # "num_nodes": int(rec.get("num_nodes")),
            "configuration": cfg_json,
        }
        for b in GRAPH_TYPES:
            item_entry[b] = handle_bundle(b)
        manifest_items.append(item_entry)

        # Write per-item spectra.csv
        spectra_csv = item_dir / "spectra.csv"
        max_len = max((len(v) for v in spectra_by_bundle.values() if v is not None), default=0)
        with spectra_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", *GRAPH_TYPES])
            for k in range(max_len):
                row = [k]
                for b in GRAPH_TYPES:
                    vals = spectra_by_bundle[b]
                    row.append(float(vals[k]) if (vals is not None and k < len(vals)) else "")
                w.writerow(row)

    # Manifest
    manifest = {"items": manifest_items} # TODO `items are redundant
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=4, ensure_ascii=False))

    return manifest


def load_dataset(in_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Loads both data and metadata (always expects experiment_metadata.json).
    Returns (data, metadata).
    """
    in_dir = Path(in_dir)
    manifest = json.loads((in_dir / "manifest.json").read_text())
    items = manifest["items"]

    exp_meta_path = in_dir / "experiment_metadata.json"
    experiment_metadata = json.loads(exp_meta_path.read_text()) if exp_meta_path.exists() else {}

    def read_item_spectra(item_dir: Path) -> dict[str, np.ndarray] | None:
        path = item_dir / "spectra.csv"
        if not path.exists():
            return None
        rows = list(csv.DictReader(path.open("r", newline="")))
        out: dict[str, list[float]] = {b: [] for b in GRAPH_TYPES}
        for row in rows:
            for b in GRAPH_TYPES:
                cell = row[b]
                if cell != "" and cell is not None:
                    out[b].append(float(cell))
        return {b: np.asarray(vals, dtype=float) for b, vals in out.items() if len(vals) > 0}

    def rebuild_bundle(meta: dict[str, Any], item_dir: Path, bundle_name: str, spectra_map: dict[str, np.ndarray] | None):
        graph_obj = _load_graph(in_dir / meta["graph_json"])
        laplacian_obj = _load_array(in_dir / meta["laplacian_npy"])
        laplacian_spectrum = None
        if spectra_map and bundle_name in spectra_map:
            laplacian_spectrum = spectra_map[bundle_name]

        # Start with the core pieces we always return
        out = {
            "graph_obj": graph_obj,
            "laplacian_obj": laplacian_obj,
            "laplacian_spectrum": laplacian_spectrum,
        }

        # Bring back any extra attributes we serialized into the manifest (already JSON-safe):
        # - metadata
        # - laplacian_pauli_repr
        # - laplacian_sparse_pauli_repr
        # - seed (if present)
        for k in ("metadata", "laplacian_pauli_repr", "laplacian_sparse_pauli_repr", "seed"):
            if k in meta:
                out[k] = meta[k]

        return out

    rebuilt: list[dict[str, Any]] = []
    for item in items:
        item_dir = in_dir / item["item_id"]
        spectra_map = read_item_spectra(item_dir)
        rebuilt.append({
            "config_index": item["config_index"],
            "configuration": item["configuration"],
            # "num_nodes": item["num_nodes"],
            **{b: rebuild_bundle(item[b], item_dir, b, spectra_map) for b in GRAPH_TYPES},
            "item_id": item["item_id"],
        })

    return rebuilt, manifest, experiment_metadata


# Optional: convenience accessor for the exported class-level metadata
def load_experiment_metadata(in_dir: str | Path) -> dict[str, Any] | None:
    path = Path(in_dir) / "experiment_metadata.json"
    return json.loads(path.read_text()) if path.exists() else None