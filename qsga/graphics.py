from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.linalg.linalg import EighResult
    from qiskit.quantum_info import SparsePauliOp
    

def obtain_math_text_hermitian_spectrum(
    eigh_result: EighResult | None = None,
    eigvals: list[float] | None = None
) -> str:
    """For a given eigenspectrum, generate a nice math LaTeX representation.
    
    Args:
        eigh_result: Result of `numpy.linalg.eigh`, containing eigenvalues and eigenvectors.
        eigvals: List of eigenvalues.

    Returns:
        str: LaTeX formatted string representing the eigenvalues, degeneracies grouped.
    """

    if eigvals is None and eigh_result is None:
        raise ValueError("Either eigvals or eigh_result must be provided")
    if eigvals is not None and eigh_result is not None:
        raise ValueError("Both eigvals and eigh_result cannot be provided simultaneously")
    if eigvals is None:
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


def sparse_pauli_op_to_latex(sparse_pauli_op: SparsePauliOp) -> str:
    """
    Convert a SparsePauliOp object to LaTeX format for nice display with IPython.Math().
    ### NOTE: THIS FUNCTION HAS BEEN COMPLETELY VIBE-CODED. ###
    
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