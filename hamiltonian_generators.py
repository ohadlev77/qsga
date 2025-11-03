from copy import deepcopy

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# Basis elements for all 2X2 matrices, building blocks for bases for all 2^nX2^n matrices
OP00 = SparsePauliOp(data=["I", "Z"], coeffs=[0.5, 0.5])
OP01 = SparsePauliOp(data=["X", "Y"], coeffs=[0.5, 0.5j])
OP10 = SparsePauliOp(data=["X", "Y"], coeffs=[0.5, -0.5j])
OP11 = SparsePauliOp(data=["I", "Z"], coeffs=[0.5, -0.5])
BASIS_OPERATORS = [OP00, OP01, OP10, OP11]


def obtain_random_pauli_strings(
    strings_length: int, # n_num_qubits
    num_strings: int = 1, # d_regularity
    basis_paulis: set[str] = {"I", "X"},
    locality: int | None = None
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
            candidate = "".join(np.random.choice(tuple(basis_paulis), size=strings_length))
        else:
            candidate_list = ["I"] * strings_length

            # Choose random positions for non-identity terms
            non_i_positions = np.random.choice(
                strings_length,
                size=min(np.random.randint(1,locality) + 1, strings_length),
                replace=False
            )

            # Fill chosen positions with random non-identity Paulis
            non_i_paulis = basis_paulis - {"I"}
            for pos in non_i_positions:
                candidate_list[pos] = np.random.choice(tuple(non_i_paulis))
            candidate = "".join(candidate_list)

        # Excluding the identity operator
        if candidate != "I" * strings_length:
            pauli_strings.add(candidate)

    return list(pauli_strings)


def obtain_ix_n_hamiltonian(
    n: int,
    d: int,
    max_locality: int | None = None
) -> SparsePauliOp:
    """Generate an IX_n Laplacian Hamiltonian over `n` qubits (2^n nodes) and `d`-regularity.

    Args:
        n: Number of qubits (log2 of number of nodes).
        d: Regularity of the graph.
        max_locality: Maximum locality of the Pauli strings used in the Hamiltonian.

    Returns:
        SparsePauliOp: The IX_n Laplacian Hamiltonian in a sparse Pauli representation.
    """

    ops = ["I" * n]
    ops += obtain_random_pauli_strings(
        strings_length=n,
        num_strings=d,
        locality=max_locality
    )

    coeffs = [d]
    coeffs += [-1 for _ in range(d)]

    return SparsePauliOp(data=ops, coeffs=coeffs)


def obtain_random_m_local_perturbation(
    m: int,
    randomly_weighted: bool = False,
    weights_range: tuple[float, float] = (0.5, 5.0),
    simplify: bool = True,
) -> SparsePauliOp:
    """TODO COMPLETE."""

    local_hilbert_dim = 2 ** m

    # Pick entris at random
    i1, i2 = np.random.choice(local_hilbert_dim, size=2, replace=False)
    entries = [(i1, i1), (i1, i2), (i2, i1), (i2, i2)]

    weights = np.array([1, -1, -1, 1])
    if randomly_weighted:
        weights = np.random.uniform(*weights_range) * weights

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


def obtain_random_perturbated_laplacian(
    skeleton_hamiltonian: SparsePauliOp,
    num_perturbations: int,
    max_perturbation_locality: int,
    random_perturbation_weights: bool = False,
    random_perturbations_scaling: bool = False,
    simplify: bool = True,
) -> SparsePauliOp:
    """TODO COMPLETE."""

    num_nodes = skeleton_hamiltonian.dim
    num_qubits = skeleton_hamiltonian.num_qubits
    perturbated_hamiltonian = deepcopy(skeleton_hamiltonian)

    # Adding perturbations
    for _ in range(num_perturbations):
        perturbation_locality = np.random.randint(max_perturbation_locality) + 1

        unscaled_perturbation = obtain_random_m_local_perturbation(
            m=perturbation_locality,
            randomly_weighted=random_perturbation_weights
        )
        scaling_dim = num_qubits - perturbation_locality

        if random_perturbations_scaling:
            threshold = np.random.randint(scaling_dim) + 1
            scaled_perturbation = SparsePauliOp("I" * threshold).tensor(unscaled_perturbation).tensor(
                SparsePauliOp("I" * (scaling_dim - threshold))
            )

            perturbated_hamiltonian += scaled_perturbation
        else:
            perturbated_hamiltonian += SparsePauliOp("I" * scaling_dim).tensor(unscaled_perturbation)
    
    if simplify:
        perturbated_hamiltonian = perturbated_hamiltonian.simplify()

    return perturbated_hamiltonian


### OBSOLETE FUNCTION BELOW ###
def obtain_random_nontrivial_laplacian(
    n_num_qubits: int,
    d_skeleton_regularity: int = 3,
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """TODO THIS FUNCTION IS OBSOLETE AND SEEMS TO PROVIDE UNVALID LAPLACIANS."""
    
    num_nodes = 2 ** n_num_qubits

    skeleton_hamiltonian = obtain_ix_n_hamiltonian(n_num_qubits, d_skeleton_regularity)
    perterbuted_hamiltonian = deepcopy(skeleton_hamiltonian)

    op_00 = SparsePauliOp(data=["II", "IZ", "ZI", "ZZ"], coeffs=[0.25 for _ in range(4)])
    op_01 = SparsePauliOp(data=["XX", "XY", "YX", "YY"], coeffs=[0.25, 0.25j, 0.25j, -0.25])
    op_10 = SparsePauliOp(data=["XX", "XY", "YX", "YY"], coeffs=[0.25, -0.25j, -0.25j, -0.25])
    op_11 = SparsePauliOp(data=["II", "IZ", "ZI", "ZZ"], coeffs=[0.25, -0.25, -0.25, 0.25])

    addition_size = 4
    scaling_paulis = obtain_random_pauli_strings(
        strings_length=n_num_qubits - 2,
        num_strings=addition_size,
        basis_paulis={"I", "X"},
        locality=2
    )

    for pauli_string in scaling_paulis:
        perterbuted_hamiltonian += SparsePauliOp("I" * (n_num_qubits - 2)).tensor(op_00)
        perterbuted_hamiltonian -= SparsePauliOp(pauli_string).tensor(op_01)
        perterbuted_hamiltonian -= SparsePauliOp(pauli_string).tensor(op_10)
        perterbuted_hamiltonian += SparsePauliOp(pauli_string).tensor(op_11)

    return skeleton_hamiltonian, perterbuted_hamiltonian


if __name__ == "__main__":
    H = obtain_ix_n_hamiltonian(n=4, d=3, max_locality=2)

    r = obtain_random_perturbated_laplacian(
        skeleton_hamiltonian=H,
        num_perturbations=4,
        max_perturbation_locality=2,
        random_perturbation_weights=True,
        random_perturbations_scaling=True,
    )

    print(r)