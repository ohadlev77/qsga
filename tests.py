import numpy as np

from hamiltonian_generators import obtain_ix_n_hamiltonian
from util import compute_eigenspectrum_ixn_laplacian


def test_compute_eigenspectrum_ixn_laplacian() -> None:
    rng = np.random.default_rng(20)
    n = 3

    laplacian = obtain_ix_n_hamiltonian(
        n=n,
        d=3,
        max_locality=2,
        pseudo_rng=rng,
    )

    analytic_eigs = compute_eigenspectrum_ixn_laplacian(laplacian)
    numeric_eigs = np.linalg.eigvalsh(laplacian.to_matrix().real)

    np.testing.assert_allclose(analytic_eigs, numeric_eigs, rtol=1)

    # last_index = 2**n - 1
    # largest_eigval_analytic = compute_eigenspectrum_ixn_laplacian(laplacian, specific_eigenvalues_indexes=[last_index])[0]
    # largest_eigval_diagonalization = numeric_eigs[-1]

    # np.testing.assert_almost_equal(largest_eigval_analytic, largest_eigval_diagonalization)


if __name__ == "__main__":
    test_compute_eigenspectrum_ixn_laplacian()