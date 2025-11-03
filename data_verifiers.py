from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from util import decompose_laplacian_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray
    

def is_valid_laplacian(matrix: NDArray[np.float64] | SparsePauliOp, is_weigthed: bool = True) -> bool:
    """Verify that `matrix` is a valid weighted/unweighted Laplacian matrix."""

    if isinstance(matrix, SparsePauliOp):
        matrix = matrix.to_matrix()

    if not np.all(np.isreal(matrix)):
        print("The matrix contains complex numbers")
        return False
    
    matrix = matrix.real
        
    if not np.all(matrix == matrix.T):
        print("The matrix is not symmetric")
        return False
    
    for row_index, row in enumerate(matrix):
        if not np.isclose(np.sum(row), 0):
            print(f"The sum of row/column {row_index} is not 0")
            return False
        
    if not is_weigthed:
        if not np.allclose(matrix % 1, 0):
            print("[Unweighted Laplacian check] Some entry is not an integer")
            return False
        
        diagonal, D, A = decompose_laplacian_matrix(matrix)
        if not np.all(diagonal >= 0):
            print("[Unweighted Laplacian check] Some diagonal entry is not a nonnegative integer")
            return False

        if not np.all((A == 0) | (A == 1)):
            print("[Unweighted Laplacian check] Some non-diagonal entry is not -1 or 0")
            return False
    
    print(f"The matrix is a valid ({'weighted' if is_weigthed else 'unweighted'}) Laplacian matrix")
    return True


def detect_array_duplications(arrays: list[NDArray]) -> set[int]:
    """For a given list of NumPy `arrays`, detect wether are there any identical arrays.
    
    Returns:
        set[int]: The indexes of the duplicated arrays (empty if no duplications).
    """

    unique_hashed_arrays = []
    duplicates_indexes = set()

    for array_index, array in enumerate(arrays):
        hashed_array = hash(array.tobytes())

        if hashed_array in unique_hashed_arrays:
            if len(duplicates_indexes) == 0:
                duplicates_indexes.add(unique_hashed_arrays.index(hashed_array))
            duplicates_indexes.add(array_index)
        else:
            unique_hashed_arrays.append(hashed_array)
    
    return duplicates_indexes