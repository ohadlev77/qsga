# Data Directory Guide (***MOSTLY GPT-GENERATED***)

This directory contains all generated experiment runs.  
Each run is stored in a **timestamped subfolder** (e.g. `2025-11-05_10-42-18_<RUN-NAME-IF-SET>/`),  
and each run folder contains several **configuration subfolders** — one per experiment.

---

## Folder Naming Convention

Each configuration folder name encodes its parameters:

`<index>-n<num_qubits>-d<skeleton_regularity>-sl<skeleton_locality>-p<num_perturbations>-pl<perturbation_locality>[-s<seed>]`

| Symbol | Field | Meaning |
|:--:|:--|:--|
| **n**  | `n_num_qubits`               | Number of qubits |
| **d**  | `d_skeleton_regularity`      | Skeleton graph regularity |
| **sl** | `max_skeleton_locality`      | Skeleton locality |
| **p**  | `num_perturbations`          | Number of perturbations |
| **pl** | `max_perturbation_locality`  | Perturbation locality |
| **s**  | `seed`                       | Random seed (optional) |

**Example:**  
`0007-n9-d4-sl3-p11-pl6-s2` → $9$ qubits, $4$-regular skeleton, $3$-local skeleton Hamiltonian, $11$ perturbations, $6$-local perturbations, seed = $2$.

---

## File Structure per Configuration

| File | Description |
|:--|:--|
| `metadata.json` | Basic configuration and runtime info |
| `spectra.csv` | Ordered Laplacian eigenvalues (one column per graph type) |
| `<graph>.graph.json` | Serialized NetworkX graph (readable JSON) |
| `<graph>.laplacian.npy` | Laplacian matrix (NumPy binary array) |


**At the run root:**

- `manifest.json` — global index of all items in the run  
- `experiment_metadata.json` — full exported configuration metadata  
  (contains `"configurations"` for easy reloading into `ExperimentConfigurations`)