GRAPH_TYPES = [
    "skeleton_graph",
    "definite_order_perturbed_graph", # PerturbationScalingMethod.LEFT
    "random_order_perturbed_graph", # PerturbationScalingMethod.RANDOM_LEFT_RIGHT
    "random_order_scrambled_perturbed_graph",  # PerturbationScalingMethod.SCRAMBLE
    "dop_like_random_graph",
    "rop_like_random_graph",
    "rop_like_random_graph_same_weights",
    "scrambled_like_random_graph",
    "scrambled_like_random_graph_same_weights"
]


OBSOLETE_GRAPHS = (
    "definite_order_perturbed_graph",
    "random_order_perturbed_graph",
    "dop_like_random_graph",
    "rop_like_random_graph",
    "scrambled_like_random_graph"
)


# {graph_type: legend_text}
GRAPHS_TO_PLOT_MAP = {
    "skeleton_graph": "Skeleton",
    "random_order_scrambled_perturbed_graph": "Randomly Perturbed",
    "scrambled_like_random_graph_same_weights": "Erdős–Rényi",
    # "scrambled_like_random_graph": "ER not same weights" # TODO NEED TO CHOOSE ONE OF THESE
}


GRAPHS_COMPARISON_PAIR = (
    "random_order_scrambled_perturbed_graph",
    "scrambled_like_random_graph_same_weights"
)