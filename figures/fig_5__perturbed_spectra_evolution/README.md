## Figure 5 — Evolution of the Perturbed Graph Laplacian with Increasing Perturbations

Figures generated from experimental run `2026-05-27_11-54-18`.

### Configuration

| Parameter | Value |
|---|---|
| Qubits $q$ | 8 ($N = 2^8 = 256$ nodes) |
| Skeleton regularity $d$ | 3 |
| Max skeleton locality $sl$ | 3 |
| Max perturbation locality $m$ | 4 |
| Perturbation weight bounds | $[0.5,\ 5]$ |
| Seed | 32 |

### Subfigure index

Each letter corresponds to a different number of perturbations applied to the skeleton Laplacian.

| Prefix | $n_p$ (formula) | $n_p$ for $q=8$ |
|---|---|---|
| A | $0$ | 0 |
| B | $\lfloor\sqrt{q}\rfloor$ | 2 |
| C | $q$ | 8 |
| D | $q^2$ | 64 |
| E | $2^q$ | 256 |

Each subfigure contains three plot types:

- `*__spectra.png` — Laplacian eigenvalue spectrum (eigenvalue index vs. eigenvalue).
- `*__laplacian.png` — Sparsity pattern of the dense Laplacian matrix.
- `*__graph.png` — Spring-layout graph drawing, nodes coloured by weighted degree.
