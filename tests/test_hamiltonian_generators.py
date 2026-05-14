import pytest
from qiskit.quantum_info import SparsePauliOp

from qsga.hamiltonian_generators import (
    embed_on_fixed_targets,
    obtain_random_pauli_strings,
    obtain_skeleton_laplacian,
    obtain_random_m_local_perturbation,
    scale_m_local_perturbation,
    obtain_random_perturbed_laplacian,
    PerturbationScalingMethod
)


# If `VERBOSE is True`, the scaling process is printed by the tests below upon `pytest -s`
VERBOSE = True


def test_embed_on_fixed_targets_basic():
    """Test basic embedding of a 2-local operator into a 4-qubit space."""
    local_op = SparsePauliOp(["XY", "IZ"], [1.0, 2.0])
    q_global_dim = 4
    targets = [3, 1]
    
    embedded_op = embed_on_fixed_targets(local_op, q_global_dim, targets)
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_basic ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print(f"Embedded Op:\n{embedded_op}")
        
    # Expected: 
    # For "XY": qubit 0 (Y) -> target 3, qubit 1 (X) -> target 1. Result: YIXI
    # For "IZ": qubit 0 (Z) -> target 3, qubit 1 (I) -> target 1. Result: ZIII
    expected_op = SparsePauliOp(["YIXI", "ZIII"], [1.0, 2.0])
    
    assert embedded_op == expected_op


def test_embed_on_fixed_targets_single_qubit():
    """Test embedding a 1-local operator into a 3-qubit space."""
    local_op = SparsePauliOp(["X", "Y", "Z"], [0.5, 1.5, 2.5])
    q_global_dim = 3
    targets = [1]
    
    embedded_op = embed_on_fixed_targets(local_op, q_global_dim, targets)
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_single_qubit ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print(f"Embedded Op:\n{embedded_op}")
        
    # Expected: 
    # For "X": qubit 0 (X) -> target 1. Result: IXI
    # For "Y": qubit 0 (Y) -> target 1. Result: IYI
    # For "Z": qubit 0 (Z) -> target 1. Result: IZI
    expected_op = SparsePauliOp(["IXI", "IYI", "IZI"], [0.5, 1.5, 2.5])
    
    assert embedded_op == expected_op


def test_embed_on_fixed_targets_identity():
    """Test embedding when the local operator contains identity."""
    local_op = SparsePauliOp(["II"], [1.0])
    q_global_dim = 2
    targets = [0, 1]
    
    embedded_op = embed_on_fixed_targets(local_op, q_global_dim, targets)
    expected_op = SparsePauliOp(["II"], [1.0])
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_identity ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print(f"Embedded Op:\n{embedded_op}")
        
    assert embedded_op == expected_op


def test_embed_on_fixed_targets_invalid_targets_length():
    """Test that an assertion error is raised if the targets length doesn't match the local operator dimension."""
    local_op = SparsePauliOp(["XYZ"], [1.0])
    q_global_dim = 5
    targets = [0, 1] # Only 2 targets for a 3-local operator
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_invalid_targets_length ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print("Expecting AssertionError to be raised.")

    with pytest.raises(AssertionError):
        embed_on_fixed_targets(local_op, q_global_dim, targets)


def test_embed_on_fixed_targets_out_of_order():
    """Test embedding when targets are provided out of order."""
    local_op = SparsePauliOp(["XYZ", "IXY"], [1.5, -2.0])
    q_global_dim = 6
    targets = [5, 1, 3]
    
    embedded_op = embed_on_fixed_targets(local_op, q_global_dim, targets)
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_out_of_order ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print(f"Embedded Op:\n{embedded_op}")
        
    expected_op = SparsePauliOp(["ZIXIYI", "YIIIXI"], [1.5, -2.0])
    assert embedded_op == expected_op


def test_embed_on_fixed_targets_dense_permutation():
    """Test embedding when local dimension equals global dimension but targets are permuted."""
    local_op = SparsePauliOp(["XYZ"], [1.0])
    q_global_dim = 3
    targets = [2, 1, 0] # Reverses the order
    
    embedded_op = embed_on_fixed_targets(local_op, q_global_dim, targets)
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_dense_permutation ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print(f"Embedded Op:\n{embedded_op}")
        
    expected_op = SparsePauliOp(["ZYX"], [1.0])
    assert embedded_op == expected_op


def test_embed_on_fixed_targets_complex_coeffs():
    """Test embedding with complex coefficients and multiple identical Paulis (e.g., transversal)."""
    local_op = SparsePauliOp(["XX", "YY", "ZZ"], [1j, -1j, 2+3j])
    q_global_dim = 5
    targets = [0, 4]
    
    embedded_op = embed_on_fixed_targets(local_op, q_global_dim, targets)
    
    if VERBOSE:
        print(f"\n--- test_embed_on_fixed_targets_complex_coeffs ---")
        print(f"Local Op:\n{local_op}")
        print(f"Targets: {targets} (Global Dim: {q_global_dim})")
        print(f"Embedded Op:\n{embedded_op}")
        
    expected_op = SparsePauliOp(["XIIIX", "YIIIY", "ZIIIZ"], [1j, -1j, 2+3j])
    assert embedded_op == expected_op


def test_obtain_random_pauli_strings():
    import numpy as np
    rng = np.random.default_rng(42)
    strings = obtain_random_pauli_strings(strings_length=3, num_strings=2, basis_paulis={"I", "X"}, locality=2, pseudo_rng=rng)
    assert len(strings) == 2
    for s in strings:
        assert len(s) == 3
        assert s != "III"
        assert set(s).issubset({"I", "X"})
        assert s.count("X") <= 2
        
    if VERBOSE:
        print(f"\n--- test_obtain_random_pauli_strings ---")
        print(f"Generated Strings: {strings}")


def test_obtain_skeleton_laplacian():
    op = obtain_skeleton_laplacian(n=3, d=2)
    assert isinstance(op, SparsePauliOp)
    assert op.num_qubits == 3
    assert len(op) == 3 # 1 identity + 2 paulies
    assert op.coeffs[0] == 2.0
    assert op.coeffs[1] == -1.0
    assert op.coeffs[2] == -1.0
    
    if VERBOSE:
        print(f"\n--- test_obtain_skeleton_laplacian ---")
        print(f"Skeleton Laplacian:\n{op}")


def test_obtain_random_m_local_perturbation():
    op = obtain_random_m_local_perturbation(m=2, simplify=True)
    assert op.num_qubits == 2
    
    if VERBOSE:
        print(f"\n--- test_obtain_random_m_local_perturbation ---")
        print(f"M-Local Perturbation:\n{op}")


def test_scale_m_local_perturbation():
    import numpy as np
    local_op = SparsePauliOp(["XY"], [1.0])
    rng = np.random.default_rng(42)
    scaled_left = scale_m_local_perturbation(local_op, q_global_dim=4, scaling_method=PerturbationScalingMethod.LEFT, pseudo_rng=rng)
    assert scaled_left == SparsePauliOp(["IIXY"], [1.0])
    
    scaled_random = scale_m_local_perturbation(local_op, q_global_dim=4, scaling_method=PerturbationScalingMethod.RANDOM_LEFT_RIGHT, pseudo_rng=rng)
    assert scaled_random.num_qubits == 4
    
    scaled_scramble = scale_m_local_perturbation(local_op, q_global_dim=4, scaling_method=PerturbationScalingMethod.SCRAMBLE, pseudo_rng=rng)
    assert scaled_scramble.num_qubits == 4
    
    if VERBOSE:
        print(f"\n--- test_scale_m_local_perturbation ---")
        print(f"Local Op:\n{local_op}")
        print(f"Scaled Left:\n{scaled_left}")
        print(f"Scaled Random:\n{scaled_random}")
        print(f"Scaled Scramble:\n{scaled_scramble}")


def test_obtain_random_perturbed_laplacian():
    skel = obtain_skeleton_laplacian(n=4, d=2)
    perturbed = obtain_random_perturbed_laplacian(
        skeleton_hamiltonian=skel,
        num_perturbations=2,
        max_perturbation_locality=2,
        perturbations_scaling_method=PerturbationScalingMethod.LEFT
    )
    assert perturbed.num_qubits == 4

    if VERBOSE:
        print(f"\n--- test_obtain_random_perturbed_laplacian ---")
        print(f"Skeleton:\n{skel}")
        print(f"Perturbed:\n{perturbed}")