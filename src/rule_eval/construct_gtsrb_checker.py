from .rule_checker import ConceptGraph, Constraint
import numpy as np

warning_indices = [21, 22, 25, 26, 27, 29, 30, 34, 35, 36]
warning_symbols_indices = [21, 22, 25, 26, 27, 29, 30]
regulatroty_signs_indices = [i for i in range(18,34) if i not in warning_indices]

def construct_full_graph(concepts_per_class_file):
    # === Implementing the Semantic Concept Graph ===

    # Initialize the Semantic ConceptGraph
    concept_graph_semantic_relation = ConceptGraph(concepts_per_class_file)

    # === Adding Nodes ===

    # Root node
    concept_graph_semantic_relation.add_concept(
        name="All concepts",
        concept_indices=list(range(43)),
    )

    # Level 1 Nodes
    concept_graph_semantic_relation.add_concept(
        name="All colors",
        concept_indices=list(range(0, 6))
    )
    # R1: Exactly one shape
    concept_graph_semantic_relation.add_concept(
        name="All shapes",
        concept_indices=list(range(6, 10)),
        constraint=[
            Constraint("exactly_one_shape", lambda x: np.sum(x) == 1)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="All symbols",
        concept_indices=list(range(10, 43)),
    )

    # Level 2 Nodes under "All colors"
    # R2: Exactly one main colour
    concept_graph_semantic_relation.add_concept(
        name="Main colors",
        concept_indices=list(range(0, 4)),
        constraint=[
            Constraint("exactly_one_main_colour", lambda x: np.sum(x) == 1)
        ]
    )
    # R3: At most one border colour
    concept_graph_semantic_relation.add_concept(
        name="Border colors",
        concept_indices=list(range(4, 6)),
        constraint=[
            Constraint("at_most_one_border_colour", lambda x: np.sum(x) <= 1)
        ]
    )

    concept_graph_semantic_relation.add_concept(
        name="General symbols",
        concept_indices=list(range(18, 34)),
    )
    concept_graph_semantic_relation.add_concept(
        name="Curve symbols",
        concept_indices=list(range(34, 37)),
    )
    concept_graph_semantic_relation.add_concept(
        name="Arrow symbols",
        concept_indices=list(range(37, 43)),
    )

    # Additional Nodes for Semantic Invariants
    # R6: At most one warning symbol
    concept_graph_semantic_relation.add_concept(
        name="Warning concepts",
        concept_indices=warning_indices,
        constraint=[
            Constraint("at_most_one_warning", lambda x: np.sum(x) <= 1)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="Regulatory signs",
        concept_indices=regulatroty_signs_indices,
    )

    # Level 3 Nodes under "Warning symbols"
    concept_graph_semantic_relation.add_concept(
        name="Curve symbols",
        concept_indices=list(range(34, 37)),
    )
    
    concept_graph_semantic_relation.add_concept(
        name="Warning symbols",
        concept_indices=warning_symbols_indices,
    )

    concept_graph_semantic_relation.add_relation(
        from_node="All concepts",
        to_node="All colors",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All concepts",
        to_node="All shapes",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All concepts",
        to_node="All symbols",
    )

    # Connect "All colors" to its children
    concept_graph_semantic_relation.add_relation(
        from_node="All colors",
        to_node="Main colors",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All colors",
        to_node="Border colors",
    )

    # Connect "All symbols" to its children
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="General symbols",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="Curve symbols",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="Arrow symbols",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="Warning concepts",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="Curve symbols",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="Warning symbols",
    )
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="Regulatory signs",
    )

    # R4: No symbols => exactly 2 colors
    complementaryset_shapes = list(range(0, 6)) + list(range(10, 43))
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="All colors",
        concept_indices=complementaryset_shapes,
        constraint=[
            Constraint("no_symbols_exactly_two_colours", lambda x: not ((np.sum(x[10:43]) == 0) and (np.sum(x[0:6]) < 2)))
        ]
    )
    
    # adding relation between arrows and overall symbols
    concept_graph_semantic_relation.add_relation(
        from_node="Arrow symbols",
        to_node="All symbols",
        concept_indices=list(range(10, 43)),
    )

    # main_color and border_color need to be different colors
    concept_graph_semantic_relation.add_relation(
        from_node="Main colors",
        to_node="Border colors",
        concept_indices=list(range(0, 6)),
    )

    # this one is commented out since it holds true for the GTSRB set 
    blue_arrow = list(range(37, 43))
    blue_arrow.append(2)
    concept_graph_semantic_relation.add_relation(
        from_node="Main colors",
        to_node="Arrow symbols",
        concept_indices=blue_arrow,
    )
    
    # R8: Warning symbol => white main colour
    warning_main_color = warning_indices + [0]
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="Main colors",
        concept_indices=warning_main_color,
        constraint=[
            Constraint("warning_implies_main_white", lambda x: False if (np.sum(x[warning_indices]) == 1 and x[0] == 0) else True)
        ]
    )
    
    # R7: Warning symbol => red border
    warning_border_color = warning_indices + [5]
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="Border colors",
        concept_indices=warning_border_color,
        constraint=[
            Constraint("warning_implies_border_red", lambda x: False if (np.sum(x[warning_indices]) == 1 and x[5] == 0) else True)
        ]
    )
    
    warning_shape = warning_indices + [7]
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="All shapes",
        concept_indices=warning_shape,
    )
    
    # R5: Warning symbol => no other symbols
    all_symbols_indices = list(range(10, 43))
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="All symbols",
        concept_indices=all_symbols_indices,
        constraint=[
            Constraint(
                "warning_sign_exclusivity",
                lambda x: not (np.sum(x[warning_indices]) == 1 and np.sum(x[10:43]) > 1)
            )
        ]
    )
    
    return concept_graph_semantic_relation

if __name__ == "__main__":
    construct_full_graph()
