""" Write a function that takes a tree as an input, expressed as Node object,
that has a children attribute that contains a list of, and it returns the
number of nodes with the same number of descendants on each of its childre.  If
a node has 0 children it's considered that its children have all the same
number of descendants.

"""

from dataclasses import dataclass
import pytest
from typing import List

@dataclass
class Node:
    children: List["Node"]


def get_number_of_balanced_nodes(root: Node) -> int:
    def dfs(node: Node) -> tuple[int, int]:
        """
        Returns:
            (descendant_count, balanced_node_count)
        """
        if not node.children:
            # Leaf node: balanced by definition
            return 0, 1

        descendant_counts = []
        balanced_count = 0

        for child in node.children:
            child_descendants, child_balanced = dfs(child)
            descendant_counts.append(child_descendants)
            balanced_count += child_balanced

        # Check if all children have the same number of descendants
        is_balanced = len(set(descendant_counts)) == 1

        # Total descendants = all children + their descendants
        total_descendants = sum(descendant_counts) + len(node.children)

        return total_descendants, balanced_count + (1 if is_balanced else 0)

    _, balanced_nodes = dfs(root)
    return balanced_nodes





def test_single_node():
    root = Node(children=[])
    assert get_number_of_balanced_nodes(root) == 1


def test_two_level_balanced():
    root = Node(children=[
        Node(children=[]),
        Node(children=[])
    ])
    assert get_number_of_balanced_nodes(root) == 3


def test_unbalanced_root():
    root = Node(children=[
        Node(children=[
            Node(children=[]),
            Node(children=[])
        ]),
        Node(children=[
            Node(children=[])
        ])
    ])
    # Leaves: 3
    # Inner nodes: both balanced
    # Root: unbalanced
    assert get_number_of_balanced_nodes(root) == 5


def test_chain_tree():
    root = Node(children=[
        Node(children=[
            Node(children=[
                Node(children=[])
            ])
        ])
    ])
    # Every node has 0 or 1 child → all balanced
    assert get_number_of_balanced_nodes(root) == 4
