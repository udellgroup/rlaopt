"""Expression tree for testing and visualization."""

from typing_extensions import Self


class ExprTree:
    """Tree representation of expression structure.

    Used for testing expression optimizations and visualizing expression structure.
    Each node has a type (class name) and zero or more children.
    """

    def __init__(self, node_type: str, *children: Self):
        """Initialize an expression tree node.

        Args:
            node_type: The expression class name.
            *children: Child ExprTree nodes.
        """
        self.node_type = node_type
        self.children = children

    def __eq__(self, other) -> bool:
        """Check structural equality of two expression trees.

        For commutative operations (AddExpression, ProductExpression), children
        can be in any order. For other operations, order matters.

        Args:
            other: Another ExprTree to compare with.

        Returns:
            bool: True if trees have the same structure and node types.
        """
        if not isinstance(other, ExprTree):
            return False
        if self.node_type != other.node_type:
            return False

        # For commutative operations, ignore child order
        if self.node_type in ("AddExpression", "ProductExpression"):
            return set(self.children) == set(other.children)

        # For other operations, order matters
        return self.children == other.children

    def __hash__(self) -> int:
        """Hash the tree for use in sets and dictionaries.

        For commutative operations, hash is order-independent.

        Returns:
            int: Hash of the tree structure.
        """
        if self.node_type in ("AddExpression", "ProductExpression"):
            # Order-independent hash for commutative operations
            return hash((self.node_type, frozenset(self.children)))
        return hash((self.node_type, self.children))

    def __repr__(self) -> str:
        """Return a code-like representation of the tree.

        Returns:
            str: String representation suitable for debugging.
        """
        if self.children:
            children_repr = ", ".join(repr(c) for c in self.children)
            return f"ExprTree({self.node_type!r}, {children_repr})"
        return f"ExprTree({self.node_type!r})"

    def __str__(self) -> str:
        """Return a pretty-printed tree visualization.

        Returns:
            str: Human-readable tree with indentation and branches.
        """
        return self._pretty(prefix="", is_root=True)

    def _pretty(self, prefix: str, is_root: bool = False, is_last: bool = True) -> str:
        """Recursively build pretty-printed tree.

        Args:
            prefix: Prefix string for current line (contains branch characters).
            is_root: Whether this is the root node.
            is_last: Whether this is the last child of its parent.

        Returns:
            str: Pretty-printed subtree.
        """
        # Current node
        if is_root:
            # Root node - no prefix or branch
            result = self.node_type + "\n"
        else:
            # Non-root node - add branch character
            branch = "└─ " if is_last else "├─ "
            result = prefix + branch + self.node_type + "\n"

        # Recursively add children
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1

            if is_root:
                # Children of root - start with empty prefix
                child_prefix = ""
            else:
                # Children of non-root get extended prefix
                extension = "   " if is_last else "│  "
                child_prefix = prefix + extension

            result += child._pretty(child_prefix, is_root=False, is_last=is_last_child)

        return result

    def depth(self) -> int:
        """Calculate the depth of the tree.

        Returns:
            int: Maximum depth from this node to any leaf.
        """
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)

    def count_nodes(self) -> int:
        """Count the total number of nodes in the tree.

        Returns:
            int: Total number of nodes (including this one).
        """
        return 1 + sum(child.count_nodes() for child in self.children)
