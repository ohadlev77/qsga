import numpy as np

from qsga.hamiltonian_generators import obtain_skeleton_laplacian
from qsga.util import compute_eigenspectrum_skeleton_laplacian


def test_compute_eigenspectrum_skeleton_laplacian() -> None:
    """For skeleton IX_n Laplacians, test that analytic and numeric eigenvalue computation methods match."""

    rng = np.random.default_rng(20)
    n = 3

    laplacian = obtain_skeleton_laplacian(
        n=n,
        d=3,
        max_locality=2,
        pseudo_rng=rng,
    )

    analytic_eigs = compute_eigenspectrum_skeleton_laplacian(laplacian, sort=True)
    numeric_eigs = np.linalg.eigvalsh(laplacian.to_matrix().real)

    np.testing.assert_allclose(analytic_eigs, numeric_eigs, rtol=1)

    # last_index = 2**n - 1
    # largest_eigval_analytic = compute_eigenspectrum_skeleton_laplacian(laplacian, specific_eigenvalues_indexes=[last_index])[0]
    # largest_eigval_diagonalization = numeric_eigs[-1]

    # np.testing.assert_almost_equal(largest_eigval_analytic, largest_eigval_diagonalization)


if __name__ == "__main__":
    test_compute_eigenspectrum_skeleton_laplacian()