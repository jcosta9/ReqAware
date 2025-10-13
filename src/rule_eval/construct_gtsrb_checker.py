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
        constraint=[
            Constraint("General concept invariant", lambda x: 3 <= np.sum(x) <= 6)
        ]
    )

    # Level 1 Nodes
    concept_graph_semantic_relation.add_concept(
        name="All colors",
        concept_indices=list(range(0, 6)),
        constraint=[
            Constraint("General color constraint", lambda x: 1 <= np.sum(x) <= 2)
        ]
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
        constraint=[
            Constraint("Symbols constraint", lambda x: np.sum(x) <= 3)
        ]
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

    # Level 2 Nodes under "All symbols"
    # concept_graph_semantic_relation.add_concept(
    #     name="Number symbols",
    #     concept_indices=list(range(10, 18)),
    #     constraint=[
    #         Constraint("Numbers constraint", lambda x: np.sum(x) <= 3)
    #     ]
    # )
    concept_graph_semantic_relation.add_concept(
        name="General symbols",
        concept_indices=list(range(18, 34)),
        constraint=[
            Constraint("General symbols constraint", lambda x: np.sum(x) <= 3)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="Curve symbols",
        concept_indices=list(range(34, 37)),
        constraint=[
            Constraint("Curve constraint", lambda x: np.sum(x) <= 1)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="Arrow symbols",
        concept_indices=list(range(37, 43)),
        constraint=[
            Constraint("Arrow constraint", lambda x: np.sum(x) <= 2)
        ]
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
        constraint=[
            Constraint("Regulatory signs constraint", lambda x: np.sum(x) <= 3)
        ]
    )

    # Level 3 Nodes under "Warning symbols"
    concept_graph_semantic_relation.add_concept(
        name="Curve symbols",
        concept_indices=list(range(34, 37)),
        constraint=[
            Constraint("Curve constraint", lambda x: np.sum(x) <= 1)
        ]
    )
    concept_graph_semantic_relation.add_concept(
        name="Warning symbols",
        concept_indices=warning_symbols_indices,  # Assuming these are already included; adjust if needed
        constraint=[
            Constraint("Warning symbols constraint", lambda x: np.sum(x) <= 1)
        ]
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
    list(range(37,43))
    concept_graph_semantic_relation.add_relation(
        from_node="Arrow symbols",
        to_node="All symbols",
        concept_indices=list(range(10, 43)),
        constraint=[
            Constraint("If arrows present no other symbols", lambda x: False if (np.sum(x[37:43]) > 0 and (np.sum(x[10:43]) != np.sum(x[37:43]))) else True)
        ]
    )

    # main_color and border_color need to be different colors
    concept_graph_semantic_relation.add_relation(
        from_node="Main colors",
        to_node="Border colors",
        concept_indices=list(range(0, 6)),
        constraint=[
            Constraint("Main and border color different", lambda x: False if (np.sum(x[0] + x[4]) == 2 or np.sum(x[1] + x[5]) == 2) else True)
        ]
    )

    # this one is commented out since it holds true for the GTSRB set 
    blue_arrow = list(range(37, 43))
    blue_arrow.append(2)
    concept_graph_semantic_relation.add_relation(
        from_node="Main colors",
        to_node="Arrow symbols",
        concept_indices=blue_arrow,
        constraint=[
            Constraint("Main blue => arrow (possible OOD)", lambda x: False if (np.sum(x[2] == 1 and np.sum(x[37:43]) == 0)) else True)
        ]
    )

    # adding the constraint that if one number is found another number needs to be present
    # concept_graph_semantic_relation.add_relation(
    #     from_node="Number symbols",
    #     to_node="Number symbols",
    #     concept_indices=list(range(10, 18)),
    #     constraint=[
    #         Constraint("One number", lambda x: False if np.sum(x[10:18]) == 1 else True)
    #     ]
    # )
    
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
        constraint=[
            Constraint("Warning => Shape Triangle", lambda x: False if (np.sum(x[warning_indices]) == 1 and x[7] == 0) else True)
        ]
    )
    concept_graph_semantic_relation.print_hierarchy()
    return concept_graph_semantic_relation

if __name__ == "__main__":
    construct_full_graph()
