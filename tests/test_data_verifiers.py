import pytest

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qsga.data_verifiers import (
    throw_laplacian_validation_exception,
    LaplacianValidationError,
    is_valid_laplacian,
    detect_array_duplications
)


VERBOSE = True


def test_throw_laplacian_validation_exception():
    with pytest.raises(LaplacianValidationError, match="test msg"):
        throw_laplacian_validation_exception("test msg")

    if VERBOSE:
        print(f"\n--- test_throw_laplacian_validation_exception ---")
        print("Successfully threw and caught LaplacianValidationError")


def test_is_valid_laplacian_valid_weighted():
    mat = np.array([
        [ 1.0, -0.5, -0.5],
        [-0.5,  1.0, -0.5],
        [-0.5, -0.5,  1.0]
    ])
    assert is_valid_laplacian(mat, is_weigthed=True)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_valid_weighted ---")
        print(f"Matrix:\n{mat}")
        print("Status: Valid Weighted Laplacian")


def test_is_valid_laplacian_sparse_pauli():
    # Matrix [[1, -1], [-1, 1]]
    # I = [[1, 0], [0, 1]], X = [[0, 1], [1, 0]]
    # I - X = [[1, -1], [-1, 1]]
    op = SparsePauliOp(["I", "X"], [1, -1])
    assert is_valid_laplacian(op, is_weigthed=True)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_sparse_pauli ---")
        print(f"Operator:\n{op}")
        print("Status: Valid Weighted Laplacian SparsePauliOp")


def test_is_valid_laplacian_invalid_complex():
    mat = np.array([
        [ 1.0+0j, -0.5+0.1j],
        [-0.5-0.1j, 1.0+0j]
    ])
    assert not is_valid_laplacian(mat, is_weigthed=True)
    with pytest.raises(LaplacianValidationError):
        is_valid_laplacian(mat, is_weigthed=True, throw_exception=True)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_invalid_complex ---")
        print(f"Matrix:\n{mat}")
        print("Status: Successfully identified invalid complex values")


def test_is_valid_laplacian_invalid_asymmetric():
    mat = np.array([
        [ 1.0, -0.2],
        [-0.8,  1.0]
    ])
    assert not is_valid_laplacian(mat, is_weigthed=True)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_invalid_asymmetric ---")
        print(f"Matrix:\n{mat}")
        print("Status: Successfully identified invalid asymmetric values")


def test_is_valid_laplacian_invalid_row_sum():
    mat = np.array([
        [ 1.0, -0.5],
        [-0.5,  2.0]
    ])
    assert not is_valid_laplacian(mat, is_weigthed=True)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_invalid_row_sum ---")
        print(f"Matrix:\n{mat}")
        print("Status: Successfully identified invalid row sums")


def test_is_valid_laplacian_valid_unweighted():
    mat = np.array([
        [ 2.0, -1.0, -1.0],
        [-1.0,  2.0, -1.0],
        [-1.0, -1.0,  2.0]
    ])
    assert is_valid_laplacian(mat, is_weigthed=False)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_valid_unweighted ---")
        print(f"Matrix:\n{mat}")
        print("Status: Valid Unweighted Laplacian")


def test_is_valid_laplacian_unweighted_invalid_non_int():
    mat = np.array([
        [ 1.5, -0.75, -0.75],
        [-0.75, 1.5, -0.75],
        [-0.75, -0.75, 1.5]
    ])
    assert not is_valid_laplacian(mat, is_weigthed=False)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_unweighted_invalid_non_int ---")


def test_is_valid_laplacian_unweighted_invalid_diagonal():
    mat = np.array([
        [-1.0,  1.0],
        [ 1.0, -1.0]
    ])
    assert not is_valid_laplacian(mat, is_weigthed=False)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_unweighted_invalid_diagonal ---")


def test_is_valid_laplacian_unweighted_invalid_non_diagonal():
    mat = np.array([
        [ 2.0, -2.0],
        [-2.0,  2.0]
    ])
    assert not is_valid_laplacian(mat, is_weigthed=False)

    if VERBOSE:
        print(f"\n--- test_is_valid_laplacian_unweighted_invalid_non_diagonal ---")


def test_detect_array_duplications():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    c = np.array([1.0, 2.0])
    d = np.array([5.0, 6.0])
    
    assert detect_array_duplications([a, b, c, d]) == {0, 2}
    assert detect_array_duplications([a, b, d]) == set()

    if VERBOSE:
        print(f"\n--- test_detect_array_duplications ---")
        print("Successfully detected duplications: {0, 2}")
