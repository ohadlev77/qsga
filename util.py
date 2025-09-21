from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING

import networkx as nx
from qiskit.quantum_info import SparsePauliOp

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.linalg.linalg import EighResult


class LaplacianDecomposition(NamedTuple):
    diagonal: NDArray[np.float64] # 1D array of length \V\
    D: NDArray[np.float64] # 2D array of dimenions \V\ X \V|
    A: NDArray[np.float64] # 2D array of dimenions \V\ X \V|

        
def decompose_laplacian_matrix(L: NDArray[np.float64]) -> LaplacianDecomposition:
    diagonal = np.diag(L)
    D = np.diag(diagonal)
    A = D - L
    
    return LaplacianDecomposition(diagonal, D, A)


def obtain_graph_from_laplacian(L: NDArray[np.float64]) -> nx.Graph:
    diagonal, D, A = decompose_laplacian_matrix(L)
    return nx.from_numpy_array(np.real(A))


def is_valid_laplacian(matrix: NDArray[np.float64], is_weigthed: bool = True) -> bool:
    """Verify that `matrix` is a valid weighted/unweighted Laplacian matrix."""

    if not np.all(np.isreal(matrix)):
        print("The matrix contains complex numbers")
        return False
        
    if not np.all(matrix == matrix.T):
        print("The matrix is not symmetric")
        return False
    
    for row_index, row in enumerate(matrix):
        if np.sum(row) != 0:
            print(f"The sum of row/column {row_index} is not 0")
            return False
        
    if not is_weigthed:
        if not np.allclose(matrix % 1, 0):
            print("[Unweighted Laplacian check] Some entry is not an integer")
            return False
        
        diagonal, D, A = decompose_laplacian_matrix(matrix)
        if not np.all(diagonal > 0):
            print("[Unweighted Laplacian check] Some diagonal entry is not a positive integer")
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


def obtain_math_text_hermitian_spectrum(eigh_result: EighResult) -> str:
    """TODO COMPLETE."""

    eigvals, eigvecs = eigh_result

    latex_text = ""
    last_eigval = None

    for index, eigval in enumerate(eigvals):
        index_string = "{" + f"{index + 1}" + "}"
        rounded_eigval = round(eigval, 10)
        text = fr"\lambda_{index_string} = {rounded_eigval}, \ "
        
        if rounded_eigval == last_eigval or last_eigval is None:
            latex_text += text
        else:
            latex_text += fr"\\ {text}"
            
        last_eigval = rounded_eigval

    return latex_text


def obtain_random_pauli_strings(
    strings_length: int, # n
    num_strings: int = 1, # d
    basis_paulis: set[str] = {"I", "X"},
) -> list[str]:
    """Return a random list of `num_strings` unique Pauli strings of length `strings_length"
    constructed from the `basis_paulis` elements, excluding the identity operator."""
    
    num_pauli_types = len(basis_paulis)
    allowed_paulis = {"I", "X", "Y", "Z"}
    max_uniqe_strings = num_pauli_types ** strings_length
    
    if not basis_paulis <= allowed_paulis:
        raise ValueError("'paulis' should be a set consists of the elements: 'I', X', 'Y', 'Z'.")
        
    if num_strings >= max_uniqe_strings:
        raise ValueError(
            f"It is not possible to create more than {max_uniqe_strings} unique Pauli strings",
            " excluding the identity, from the elements provided"
        )
    
    # Using a set here to exclude duplications of Pauli strings
    pauli_strings = set()
    
    while len(pauli_strings) < num_strings:
        candidate = "".join(np.random.choice(tuple(basis_paulis), size=strings_length))
        
        # Excluding the identity operator
        if candidate != "I" * strings_length:
            pauli_strings.add(candidate)

    return list(pauli_strings)


def obtain_ix_n_hamiltonian(n: int, d: int) -> SparsePauliOp:
    """TODO COMPLETE."""

    ops = ["I" * n]
    ops += obtain_random_pauli_strings(
        strings_length=n,
        num_strings=d,
    )

    coeffs = [d]
    coeffs += [-1 for _ in range(d)]

    return SparsePauliOp(data=ops, coeffs=coeffs)


def sparse_pauli_op_to_latex(sparse_pauli_op: SparsePauliOp) -> str:
    """
    Convert a SparsePauliOp object to LaTeX format for nice display with IPython.Math().
    
    Args:
        sparse_pauli_op: A SparsePauliOp object from Qiskit
        
    Returns:
        str: LaTeX formatted string representing the Pauli sum
    """
    
    # Get the Pauli strings and coefficients
    pauli_list = sparse_pauli_op.paulis
    coeffs = sparse_pauli_op.coeffs
    
    latex_terms = []
    
    for i, (pauli, coeff) in enumerate(zip(pauli_list, coeffs)):
        # Handle coefficient formatting
        if coeff.imag == 0:
            # Real coefficient
            coeff_val = coeff.real
        else:
            # Complex coefficient
            if coeff.real == 0:
                if coeff.imag == 1:
                    coeff_str = "i"
                elif coeff.imag == -1:
                    coeff_str = "-i"
                else:
                    coeff_str = f"{coeff.imag:g}i"
            else:
                real_part = f"{coeff.real:g}" if coeff.real != 0 else ""
                imag_part = coeff.imag
                if imag_part == 1:
                    imag_str = "+i" if coeff.real != 0 else "i"
                elif imag_part == -1:
                    imag_str = "-i"
                else:
                    imag_str = f"{imag_part:+g}i" if coeff.real != 0 else f"{imag_part:g}i"
                coeff_str = real_part + imag_str
        
        # For real coefficients, format nicely
        if coeff.imag == 0:
            if coeff_val == 1:
                coeff_str = ""
            elif coeff_val == -1:
                coeff_str = "-"
            elif coeff_val == 0:
                continue  # Skip zero terms
            else:
                coeff_str = f"{coeff_val:g}"
        
        # Convert Pauli string to LaTeX
        pauli_str = str(pauli)
        latex_pauli = ""
        
        for char in pauli_str:
            if char == 'I':
                latex_pauli += "I"
            elif char in ['X', 'Y', 'Z']:
                latex_pauli += char
        
        # Combine coefficient and Pauli string
        if coeff_str == "":
            term = latex_pauli if latex_pauli != "" else "I"
        elif coeff_str == "-":
            term = f"-{latex_pauli}" if latex_pauli != "" else "-I"
        else:
            if latex_pauli == "":
                term = coeff_str
            else:
                # Add space or multiplication symbol between coefficient and Pauli
                if coeff_str.replace("-", "").replace("+", "").replace("i", "").replace(".", "").isdigit() or "i" in coeff_str:
                    term = f"{coeff_str} \\, {latex_pauli}"
                else:
                    term = f"{coeff_str}{latex_pauli}"
        
        # Handle signs for terms after the first
        if i == 0:
            latex_terms.append(term)
        else:
            if term.startswith('-'):
                latex_terms.append(f" - {term[1:]}")
            else:
                latex_terms.append(f" + {term}")
    
    # Join all terms
    if not latex_terms:
        return "0"
    
    latex_str = "".join(latex_terms)
    
    # Clean up any double spaces or formatting issues
    latex_str = latex_str.replace("  ", " ").strip()
    
    return latex_str