import pytest
from collections import namedtuple

from qiskit.quantum_info import SparsePauliOp

from qsga.graphics import (
    obtain_math_text_hermitian_spectrum,
    sparse_pauli_op_to_latex
)


VERBOSE = True


def test_obtain_math_text_hermitian_spectrum_with_eigvals():
    eigvals = [1.0, 1.0, 2.5]
    result = obtain_math_text_hermitian_spectrum(eigvals=eigvals)
    
    # First eigenvalue
    assert r"\lambda_{1} = 1.0, \ " in result
    # Second eigenvalue (degenerate, shouldn't start new line)
    assert r"\lambda_{2} = 1.0, \ " in result
    # Third eigenvalue (starts new line because it's different)
    assert r"\\ \lambda_{3} = 2.5, \ " in result

    if VERBOSE:
        print(f"\n--- test_obtain_math_text_hermitian_spectrum_with_eigvals ---")
        print(f"Input: {eigvals}\nOutput: {result}")


def test_obtain_math_text_hermitian_spectrum_with_eigh_result():
    EighResult = namedtuple('EighResult', ['eigenvalues', 'eigenvectors'])
    eigh_res = EighResult([0.0, 1.2], None)
    result = obtain_math_text_hermitian_spectrum(eigh_result=eigh_res)
    
    assert r"\lambda_{1} = 0.0, \ " in result
    assert r"\\ \lambda_{2} = 1.2, \ " in result

    if VERBOSE:
        print(f"\n--- test_obtain_math_text_hermitian_spectrum_with_eigh_result ---")
        print(f"Input: {eigh_res.eigenvalues}\nOutput: {result}")


def test_obtain_math_text_hermitian_spectrum_exceptions():
    with pytest.raises(ValueError):
        obtain_math_text_hermitian_spectrum()
    
    with pytest.raises(ValueError):
        obtain_math_text_hermitian_spectrum(eigh_result=True, eigvals=[1])

    if VERBOSE:
        print(f"\n--- test_obtain_math_text_hermitian_spectrum_exceptions ---")
        print(f"Correctly raised ValueErrors for invalid inputs.")


def test_sparse_pauli_op_to_latex():
    # Simple single term
    op1 = SparsePauliOp(["X"], [1])
    assert sparse_pauli_op_to_latex(op1) == "X"
    
    # Negative coefficient
    op2 = SparsePauliOp(["Y"], [-1])
    assert sparse_pauli_op_to_latex(op2) == "-Y"
    
    # Real coefficient
    op3 = SparsePauliOp(["Z"], [2.5])
    assert sparse_pauli_op_to_latex(op3) == "2.5 \\, Z"
    
    # Multiple terms
    op4 = SparsePauliOp(["XX", "YY"], [1.0, -2.0])
    latex4 = sparse_pauli_op_to_latex(op4)
    assert "XX" in latex4
    assert "- 2 \\, YY" in latex4 or "- 2.0 \\, YY" in latex4
    
    # Complex coefficient
    op5 = SparsePauliOp(["X"], [1j])
    assert sparse_pauli_op_to_latex(op5) == "i \\, X"
    
    # Complex coefficient negative
    op6 = SparsePauliOp(["Y"], [-1j])
    assert sparse_pauli_op_to_latex(op6) == "-i \\, Y"
    
    # Mixed complex
    op7 = SparsePauliOp(["Z"], [1+2j])
    assert sparse_pauli_op_to_latex(op7) == "1+2i \\, Z"
    
    # Zero term
    op8 = SparsePauliOp(["I"], [0])
    assert sparse_pauli_op_to_latex(op8) == "0"

    if VERBOSE:
        print(f"\n--- test_sparse_pauli_op_to_latex ---")
        ops = [op1, op2, op3, op4, op5, op6, op7, op8]
        for op in ops:
            print(f"Input Operator: {op}")
            print(f"LaTeX string: {sparse_pauli_op_to_latex(op)}\n")
