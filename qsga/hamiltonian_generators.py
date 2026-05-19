from __future__ import annotations
from typing import TYPE_CHECKING
from copy import deepcopy
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import SparsePauliOp

from qsga.util import time_it

if TYPE_CHECKING:
    from numpy.random import Generator


# Basis elements for all 2X2 matrices, building blocks for bases of all 2^n X 2^n matrices
OP00 = SparsePauliOp(data=["I", "Z"], coeffs=[0.5, 0.5]) # |0><0|
OP01 = SparsePauliOp(data=["X", "Y"], coeffs=[0.5, 0.5j]) # |0><1|
OP10 = SparsePauliOp(data=["X", "Y"], coeffs=[0.5, -0.5j]) # |1><0|
OP11 = SparsePauliOp(data=["I", "Z"], coeffs=[0.5, -0.5]) # |1><1|
BASIS_OPERATORS = [OP00, OP01, OP10, OP11]


class PerturbationScalingMethod(Enum):
    """Enum for perturbation scaling methods."""

    LEFT = auto()
    """Tensoring the local perturbation with identity operators from the left,
    for example: XY -> IIIXY
    """

    RANDOM_LEFT_RIGHT = auto()
    """Randomly tensor the local perturbation with identity operators from the left and right,
    for example: XY -> IIXYI
    """

    SCRAMBLE = auto()
    """Randomly interleaving the local perturbation with identity operators from left, right and inbetween,
    for example: XY -> IYIXII
    """


def obtain_random_pauli_strings(
    strings_length: int, # n_num_qubits (q)
    num_strings: int = 1, # d_regularity (d)
    basis_paulis: set[str] = {"I", "X"},
    locality: int | None = None,
    pseudo_rng: Generator = np.random.default_rng() # No predefined seed as default
) -> list[str]:
    """Return a random list of `num_strings` unique Pauli strings of length `strings_length"
    constructed from the `basis_paulis` elements, excluding the identity operator.
    If locality is specified, each string will have at most `locality` non-identity terms.
    
    Args:
        strings_length: Length of each Pauli string.
        num_strings: Number of unique Pauli strings to generate.
        basis_paulis: Set of Pauli elements to use in string construction.
        locality: Maximum number of non-identity terms in each string.

    Returns:
        list[str]: List of unique Pauli strings of length `strings_length` over the specified `basis_paulis`.
    """

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
        if locality is None:
            candidate = "".join(pseudo_rng.choice(tuple(basis_paulis), size=strings_length))
        else:
            candidate_list = ["I"] * strings_length

            # Choose random positions for non-identity terms
            non_i_positions = pseudo_rng.choice(
                strings_length,
                size=min(pseudo_rng.integers(1, locality) + 1, strings_length),
                replace=False
            )

            # Fill chosen positions with random non-identity Paulis
            non_i_paulis = basis_paulis - {"I"}
            for pos in non_i_positions:
                candidate_list[pos] = pseudo_rng.choice(tuple(non_i_paulis))
            candidate = "".join(candidate_list)

        # Excluding the identity operator
        if candidate != "I" * strings_length:
            pauli_strings.add(candidate)

    return list(pauli_strings)


@time_it
def obtain_skeleton_laplacian(
    n: int,
    d: int,
    max_locality: int | None = None,
    pseudo_rng: Generator = np.random.default_rng() # No predefined seed as default
) -> SparsePauliOp:
    """Generate a Skeleton Laplacian Hamiltonian over `n` qubits (2^n nodes) and `d`-regularity.

    Args:
        n: Number of qubits (log2 of number of nodes).
        d: Regularity of the graph.
        max_locality: Maximum locality of the Pauli strings used in the Hamiltonian.

    Returns:
        SparsePauliOp: The Skeleton Laplacian Hamiltonian in a sparse Pauli representation.
    """

    ops = ["I" * n]
    ops += obtain_random_pauli_strings(
        strings_length=n,
        num_strings=d,
        locality=max_locality,
        pseudo_rng=pseudo_rng
    )

    coeffs = [d]
    coeffs += [-1 for _ in range(d)]

    return SparsePauliOp(data=ops, coeffs=coeffs)


def obtain_random_m_local_perturbation(
    m: int,
    weights_range: tuple[float, float] | None = None,
    simplify: bool = True,
    pseudo_rng: Generator = np.random.default_rng() # No predefined seed as default
) -> SparsePauliOp:
    """Generate a random $m$-local perturbation as a `SparsePauliOp`.

    Creates a perturbation operator acting on $m$ qubits.

    Args:
        m: Number of qubits the perturbation acts on (locality).
        weights_range: Optional tuple (min, max) for scaling the perturbation weights.
        simplify: Whether to simplify (= merge identical Pauli terms) the resulting SparsePauliOp. Defaults to True.
        pseudo_rng: Random number generator instance.

    Returns:
        SparsePauliOp: The m-local perturbation Hamiltonian.
    """

    local_hilbert_dim = 2 ** m

    # Pick entries at random
    i1, i2 = pseudo_rng.choice(local_hilbert_dim, size=2, replace=False)
    entries = [(i1, i1), (i1, i2), (i2, i1), (i2, i2)]

    weights = np.array([1, -1, -1, 1])
    if weights_range is not None:
        weights = pseudo_rng.uniform(*weights_range) * weights

    perturbation_hamiltonian = SparsePauliOp(["I" * m], [0])
    for entry_index, entry in enumerate(entries):
        binary_col: str = bin(entry[0])[2:].zfill(m)
        binary_row: str = bin(entry[1])[2:].zfill(m)

        entry_op = SparsePauliOp("")
        for bit_row, bit_col in zip(binary_row, binary_col):
            basis_operator_index = int(bit_row + bit_col, 2)
            new_op = deepcopy(BASIS_OPERATORS[basis_operator_index])

            entry_op = entry_op.tensor(new_op)

        perturbation_hamiltonian += weights[entry_index] * entry_op

    if simplify:
        perturbation_hamiltonian = perturbation_hamiltonian.simplify()

    return perturbation_hamiltonian


def scale_m_local_perturbation(
    m_local_perturbation: SparsePauliOp | NDArray[np.float64],
    q_global_dim: int,
    scaling_method: PerturbationScalingMethod,
    simplify: bool = True,
    pseudo_rng: Generator = np.random.default_rng() # No predefined seed as default
) -> SparsePauliOp:
    """Scales an m-local perturbation to act on q qubits.

    Args:
        m_local_perturbation: The m-local perturbation operator.
        q_global_dim: The number of qubits in the global Hilbert space.
        scaling_method: The method to use for scaling the perturbation.
        simplify: Whether to simplify the resulting SparsePauliOp (cancel and merge identical Pauli terms).
        pseudo_rng: Random number generator instance.

    Returns:
        SparsePauliOp: The scaled perturbation operator.
    """

    if isinstance(m_local_perturbation, np.ndarray):
        m_local_perturbation = SparsePauliOp.from_operator(m_local_perturbation)
    elif not isinstance(m_local_perturbation, SparsePauliOp):
        raise ValueError("Only NumPy array or SparsePauliOp types are allowed for `m_local_perturbation`.")

    m_local_dim = m_local_perturbation.num_qubits
    scaling_dim = q_global_dim - m_local_dim

    if scaling_method == PerturbationScalingMethod.LEFT:
        scaled_perturbation = SparsePauliOp("I" * scaling_dim).tensor(m_local_perturbation)

    elif scaling_method == PerturbationScalingMethod.RANDOM_LEFT_RIGHT:        
        # NO BIAS
        threshold = pseudo_rng.integers(scaling_dim + 1)

        # EXPONENTIAL DECAY BIAS
        # values = np.arange(scaling_dim + 1)
        # beta = 0.75  # bigger = more biased to small indices
        # probs = np.exp(-beta * values)
        # probs = probs / probs.sum()
        # probs = probs[::-1]

        # print(probs)
        # threshold = pseudo_rng.choice(values, p=probs)
        # print(f"{threshold=}")

        scaled_perturbation = SparsePauliOp("I" * threshold).tensor(m_local_perturbation).tensor(
            SparsePauliOp("I" * (scaling_dim - threshold))
        )
    
    elif scaling_method == PerturbationScalingMethod.SCRAMBLE:
        # No bias
        values = np.arange(q_global_dim)
        probs = np.ones(q_global_dim) / q_global_dim

        # LINEAR BIAS
        # values = np.arange(num_qubits)
        # probs = (num_qubits - values) / num_qubits
        # probs = probs / sum(probs)

        # POWER LAW BIAS
        # values = np.arange(num_qubits)
        # alpha = 3.5  # increase this for stronger bias
        # probs = (num_qubits - values) ** alpha
        # probs = probs / probs.sum()
        # probs = probs[::-1]

        # EXPONENTIAL DECAY BIAS
        # values = np.arange(num_qubits)
        # beta = 0.75  # bigger = more biased to small indices
        # probs = np.exp(-beta * values)
        # probs = probs / probs.sum()
        # probs = probs[::-1]

        # print(probs)
        # print(sum(probs))

        scaled_perturbation = embed_on_fixed_targets(
            local_op=m_local_perturbation,
            q_global_dim=q_global_dim,
            targets=pseudo_rng.choice(
                values,
                size=m_local_dim,
                p=probs,
                replace=False
            )
        )

    else:
        raise ValueError("NOT ALLOWED VALUE FOR `scaling_method`.")
    
    if simplify:
        scaled_perturbation = scaled_perturbation.simplify()

    return scaled_perturbation


@time_it
def obtain_random_perturbed_laplacian(
    skeleton_hamiltonian: SparsePauliOp, # L_s
    num_perturbations: int, # n_p
    max_perturbation_locality: int, # m
    random_perturbation_weights_bounds: tuple[float, float] | None = None,
    perturbations_scaling_method: PerturbationScalingMethod = PerturbationScalingMethod.SCRAMBLE,
    simplify: bool = True,
    pseudo_rng: Generator = np.random.default_rng() # No predefined seed as default
) -> SparsePauliOp:
    """Generate a random perturbed Laplacian Hamiltonian by applying multiple
    local perturbations to a Skeleton Laplacian Hamiltonian.

    Args:
        skeleton_hamiltonian: The base Hamiltonian to perturb.
        num_perturbations: The number of perturbations to apply.
        max_perturbation_locality: The maximum locality of each perturbation.
        random_perturbation_weights_bounds: Optional bounds for the perturbation weights.
        random_perturbations_scaling: Whether to apply random scaling to the perturbations (when `True`, default),
        or definite scaling (when `False`) = all identities applied from the left.
        simplify: Whether to simplify (= merge identical Pauli terms) the resulting `SparsePauliOp`. Defaults to True.
        pseudo_rng: Random number generator instance.

    Returns:
        SparsePauliOp: The resulting perturbed Laplacian Hamiltonian.
    """

    num_qubits = skeleton_hamiltonian.num_qubits
    perturbed_hamiltonian = deepcopy(skeleton_hamiltonian)

    # Adding perturbations
    for _ in range(num_perturbations):

        # TODO RESOLVE `m` IS MAX LOCALITY OR EXACT LOCALITY
        # perturbation_locality = pseudo_rng.integers(max_perturbation_locality) + 1 # m = MAX LOCALITY
        perturbation_locality = max_perturbation_locality # m = EXACT LOCALITY

        unscaled_perturbation = obtain_random_m_local_perturbation(
            m=perturbation_locality,
            weights_range=random_perturbation_weights_bounds,
            simplify=simplify,
            pseudo_rng=pseudo_rng
        )

        scaled_perturbation = scale_m_local_perturbation(
            m_local_perturbation=unscaled_perturbation,
            q_global_dim=num_qubits,
            scaling_method=perturbations_scaling_method,
            simplify=simplify,
            pseudo_rng=pseudo_rng
        )
        perturbed_hamiltonian += scaled_perturbation
    
    if simplify:
        perturbed_hamiltonian = perturbed_hamiltonian.simplify()

    return perturbed_hamiltonian


def embed_on_fixed_targets(local_op: SparsePauliOp, q_global_dim: int, targets: list[int]):
    """Embeds an m-local Pauli operator on q qubits on a specific set of m target qubits.

    Args:
        local_op: The m-local Pauli operator to embed.
        q_global_dim: The number of qubits in the global Hilbert space.
        targets: The indices of the target qubits.

    Returns:
        SparsePauliOp: The embedded Pauli operator.
    """
    
    m_local_dim = len(local_op.paulis.to_labels()[0])
    assert len(targets) == m_local_dim

    new_labels, new_coeffs = [], []
    for label, coeff in zip(local_op.paulis.to_labels(), local_op.coeffs):

        big = ["I"] * q_global_dim
        # Qiskit: rightmost char is qubit 0 (of the SMALL op)
        for small_q in range(m_local_dim):
            p = label[-1 - small_q]
            big[targets[small_q]] = p

        new_labels.append("".join(reversed(big)))  # back to Qiskit label order
        new_coeffs.append(coeff)

    return SparsePauliOp(new_labels, new_coeffs)