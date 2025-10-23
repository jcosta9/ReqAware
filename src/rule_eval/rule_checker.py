import csv
from typing import Callable
import numpy as np
import networkx as nx




class Constraint:
    # Since the proposed constraints are simple boolean functions, we can use a lambda functions to define and check them
    def __init__(self, name: str, invariant: Callable):
        self.name = name
        self.invariant = (
            invariant  # The actual constraint to check written as a callable function
        )

    def check(self, concept_vector: list) -> bool:
        return self.invariant(concept_vector)


class ConceptGraph:
    """Graph-based structure where edges represent constraints."""

    def __init__(self, concepts_per_class_file):
        self.graph = nx.DiGraph()

        # creating the concept dictionary for mapping between index in concept vector and concept name
        self.concept_dict = {}
        with open(
            concepts_per_class_file,
            mode="r",
        ) as file:
            reader = csv.reader(file)
            header = next(reader)
            for index, concept in enumerate(header[2:]):
                self.concept_dict[concept] = index

    def add_concept(
        self,
        name: str,
        concept_indices: list,
        constraint: list = None,
        print_hierarchy=False,
    ):
        """Adds a node with associated concept indices and by default prints the new hierarchy."""
        self.graph.add_node(
            name, concept_indices=concept_indices, constraint=constraint
        )
        if print_hierarchy:
            self.print_hierarchy()

    def add_relation(
        self,
        from_node: str,
        to_node: str,
        concept_indices: list = None,
        constraint: list[Constraint] = None,
    ):
        """Adds a directed edge with an associated constraint."""
        if from_node not in self.graph or to_node not in self.graph:
            raise ValueError(
                "Both nodes must exist in the graph before adding an edge."
            )
        self.graph.add_edge(
            from_node, to_node, concept_indices=concept_indices, constraint=constraint
        )

    def check_concept_vector(
        self, concept_vector: np.ndarray, verbose=False, early_stop=False
    ) -> list:
        """
        Traverses the graph and checks each edge's constraint.
        Returns a list of violated constraints.
        """
        if not isinstance(concept_vector, np.ndarray):
            raise TypeError("Concept vector must be a NumPy array.")
        if concept_vector.dtype not in [int, np.int32, np.int64]:
            raise ValueError("Concept vector must be of integer type.")
        max_index = max(
            index
            for node in self.graph.nodes
            for index in self.graph.nodes[node]["concept_indices"]
        )
        if len(concept_vector) < max_index + 1:
            raise ValueError(
                "Concept vector is shorter than required by concept indices."
            )
        # TODO change this list to extent the list by returning values and not editing the list by reference
        violated_constraints = []
        for node in self.graph.nodes:
            # checking node constraint
            node_indices = self.graph.nodes[node]["concept_indices"]
            relevant_concepts = self.get_relevant_concepts(node_indices, concept_vector)
            constraints = self.graph.nodes[node].get("constraint")
            if constraints:
                for constraint in constraints:
                    self.validate_constraint(
                        concept_vector,
                        verbose,
                        violated_constraints,
                        relevant_concepts,
                        constraint,
                    )
            if violated_constraints and early_stop:
                break
            # checking edge constraints
            for _, _, data in self.graph.out_edges(node, data=True):
                constraints = data.get("constraint")
                if constraints:
                    edge_relevant_concepts = self.get_relevant_concepts(
                        data.get("concept_indices"), concept_vector
                    )
                    for constraint in constraints:
                        self.validate_constraint(
                            concept_vector,
                            verbose,
                            violated_constraints,
                            edge_relevant_concepts,
                            constraint,
                        )
                if violated_constraints and early_stop:
                    break
        return violated_constraints

    def validate_constraint(
        self,
        concept_vector,
        verbose,
        violated_constraints,
        relevant_concepts,
        constraint,
    ):
        if not constraint.check(relevant_concepts):
            violated_constraints.append(
                {
                    "constraint": constraint.name,
                }
            )
            if verbose:
                print(f"Constraint violated: {constraint.name}")
                violating_indices = np.where(relevant_concepts)[0]
                violating_names = [
                    name
                    for name, index in CONCEPT_DICT.items()
                    if index in violating_indices and concept_vector[index]
                ]
                print(f"Violating concept names: {violating_names}")
        return violated_constraints

    def get_relevant_concepts(
        self, concept_indices: list, concept_vector: np.ndarray
    ) -> np.ndarray:
        """Returns the relevant concepts for a given vector with concept indices"""
        concept_vector_length = len(concept_vector)
        vector = [0] * concept_vector_length
        for index in concept_indices:
            if index < concept_vector_length:
                vector[index] = 1
        return np.array(vector) & concept_vector

    def visualize_graph(self, save_path=None):
        """Optional: Visualize the concept graph."""
        import matplotlib.pyplot as plt
        import networkx as nx

        pos = nx.spring_layout(self.graph)

        # Prepare edge labels
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            constraints = data.get("constraint", [])
            if constraints:
                # Extract all constraint names and join them with commas
                labels = ", ".join([constraint.name for constraint in constraints])
            else:
                labels = ""
            edge_labels[(u, v)] = labels

        # Draw nodes and edges
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=2000,
            arrows=True,
        )

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_color="red"
        )

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def print_hierarchy(self, root=None, level=0, visited=None):
        if visited is None:
            visited = set()
        if root is None:
            root = [n for n, d in self.graph.in_degree() if d == 0]
            root = root[0]
        visited.add(root)
        print("   " * level + "- " + root)
        for child in self.graph.successors(root):
            if child not in visited:
                self.print_hierarchy(child, level + 1, visited)
