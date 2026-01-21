from __future__ import annotations
from typing import TYPE_CHECKING
from copy import deepcopy

import numpy as np
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from numpy.random import Generator


# Basis elements for all 2X2 matrices, building blocks for bases of all 2^n X 2^n matrices
OP00 = SparsePauliOp(data=["I", "Z"], coeffs=[0.5, 0.5]) # |0><0|
OP01 = SparsePauliOp(data=["X", "Y"], coeffs=[0.5, 0.5j]) # |0><1|
OP10 = SparsePauliOp(data=["X", "Y"], coeffs=[0.5, -0.5j]) # |1><0|
OP11 = SparsePauliOp(data=["I", "Z"], coeffs=[0.5, -0.5]) # |1><1|
BASIS_OPERATORS = [OP00, OP01, OP10, OP11]


def obtain_random_pauli_strings(
    strings_length: int, # n_num_qubits
    num_strings: int = 1, # d_regularity
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


def obtain_random_perturbed_laplacian(
    skeleton_hamiltonian: SparsePauliOp,
    num_perturbations: int,
    max_perturbation_locality: int,
    random_perturbation_weights_bounds: tuple[float, float] | None = None,
    random_perturbations_scaling: bool = False,
    simplify: bool = True,
    pseudo_rng: Generator = np.random.default_rng() # No predefined seed as default
) -> SparsePauliOp:
    """Generate a random perturbed Laplacian Hamiltonian by applying multiple
    local perturbations to a skeleton Hamiltonian.

    Args:
        skeleton_hamiltonian: The base Hamiltonian to perturb.
        num_perturbations: The number of perturbations to apply.
        max_perturbation_locality: The maximum locality of each perturbation.
        random_perturbation_weights_bounds: Optional bounds for the perturbation weights.
        random_perturbations_scaling: Whether to apply scaling to the perturbations.
        simplify: Whether to simplify (= merge identical Pauli terms) the resulting SparsePauliOp. Defaults to True.
        pseudo_rng: Random number generator instance.

    Returns:
        SparsePauliOp: The resulting perturbed Laplacian Hamiltonian.
    """

    num_qubits = skeleton_hamiltonian.num_qubits
    perturbed_hamiltonian = deepcopy(skeleton_hamiltonian)

    # Adding perturbations
    for _ in range(num_perturbations):
        perturbation_locality = pseudo_rng.integers(max_perturbation_locality) + 1

        unscaled_perturbation = obtain_random_m_local_perturbation(
            m=perturbation_locality,
            weights_range=random_perturbation_weights_bounds,
            simplify=simplify,
            pseudo_rng=pseudo_rng
        )
        scaling_dim = num_qubits - perturbation_locality

        # TODO CAN WE FIND A WAY TO BREAK THE PERTURBATION STRING ITSELF INTO DISTINCT PIECES?
        if random_perturbations_scaling:
            threshold = pseudo_rng.integers(scaling_dim + 1)
            scaled_perturbation = SparsePauliOp("I" * threshold).tensor(unscaled_perturbation).tensor(
                SparsePauliOp("I" * (scaling_dim - threshold))
            )

            perturbed_hamiltonian += scaled_perturbation
        else:
            perturbed_hamiltonian += SparsePauliOp("I" * scaling_dim).tensor(unscaled_perturbation)
    
    if simplify:
        perturbed_hamiltonian = perturbed_hamiltonian.simplify()

    return perturbed_hamiltonian


if __name__ == "__main__":
    H = obtain_skeleton_laplacian(n=4, d=3, max_locality=2)

    r = obtain_random_perturbed_laplacian(
        skeleton_hamiltonian=H,
        num_perturbations=4,
        max_perturbation_locality=2,
        random_perturbation_weights=True,
        random_perturbations_scaling=True,
    )

    print(r)