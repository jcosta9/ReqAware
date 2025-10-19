from .rule_checker import ConceptGraph, Constraint
import numpy as np

warning_indices = [21, 22, 25, 26, 27, 29, 30, 34, 35, 36]
warning_symbols_indices = [21, 22, 25, 26, 27, 29, 30]
regulatroty_signs_indices = [i for i in range(18,34) if i not in warning_indices]

def construct_full_graph():
    # === Implementing the Semantic Concept Graph ===

    # Initialize the Semantic ConceptGraph
    concept_graph_semantic_relation = ConceptGraph()

    # === Adding Nodes ===

    # Root node
    concept_graph_semantic_relation.add_concept(
        name="All concepts",
        concept_indices=list(range(43)),
    )

    # Level 1 Nodes
    concept_graph_semantic_relation.add_concept(
        name="All colors",
        concept_indices=list(range(0, 6)),
    )
    concept_graph_semantic_relation.add_concept(
        name="All shapes",
        concept_indices=list(range(6, 10)),
        constraint=[
            Constraint("Shape constraint", lambda x: np.sum(x) == 1)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="All symbols",
        concept_indices=list(range(10, 43)),
    )

    # Level 2 Nodes under "All colors"
    concept_graph_semantic_relation.add_concept(
        name="Main colors",
        concept_indices=list(range(0, 4)),
        constraint=[
            Constraint("Main color constraint", lambda x: np.sum(x) == 1)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="Border colors",
        concept_indices=list(range(4, 6)),
        constraint=[
            Constraint("Border color constraint", lambda x: np.sum(x) <= 1)
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
    concept_graph_semantic_relation.add_concept(
        name="Warning concepts",
        concept_indices=warning_indices,
        constraint=[
            Constraint("Warning concepts constraint", lambda x: np.sum(x) <= 1)
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
        concept_indices=warning_symbols_indices,  # Assuming these are already included; adjust if needed
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
    # concept_graph_semantic_relation.add_relation(
    #     from_node="All symbols",
    #     to_node="Number symbols",
    # )
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

    # symbols -> color relation
    # this triggers if there are no symbols and the number of colors is smaller than 2
    # turns out, this case is already covered by the general concept invariant, but it might be useful for failure analysis
    complementaryset_shapes = list(range(0, 6)) + list(range(10, 43))
    concept_graph_semantic_relation.add_relation(
        from_node="All symbols",
        to_node="All colors",
        concept_indices=complementaryset_shapes,
        constraint=[
            Constraint("No symbols => 2 colors", lambda x: not ((np.sum(x[10:43]) == 0) and (np.sum(x[0:6]) < 2)))
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
    
    # adding relation between warning symbols, shpe and color
    warning_main_color = warning_indices + [0]
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="Main colors",
        concept_indices=warning_main_color,
        constraint=[
            Constraint("Warning => Main color white", lambda x: False if (np.sum(x[warning_indices]) == 1 and x[0] == 0) else True)
        ]
    )
    warning_border_color = warning_indices + [5]
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="Border colors",
        concept_indices=warning_border_color,
        constraint=[
            Constraint("Warning => Border color red", lambda x: False if (np.sum(x[warning_indices]) == 1 and x[5] == 0) else True)
        ]
    )
    warning_shape = warning_indices + [7]
    concept_graph_semantic_relation.add_relation(
        from_node="Warning concepts",
        to_node="All shapes",
        concept_indices=warning_shape,
    )
    #concept_graph_semantic_relation.print_hierarchy()
    return concept_graph_semantic_relation

if __name__ == "__main__":
    construct_full_graph()
