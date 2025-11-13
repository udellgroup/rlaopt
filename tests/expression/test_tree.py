"""Tests for Tree class."""

import pytest

from rlaopt.expression.tree import ExprTree


@pytest.fixture
def leaf_node():
    """Create a simple leaf node."""
    return ExprTree("LeafA")


@pytest.fixture
def simple_tree():
    """Create a simple tree with one parent and two children."""
    left = ExprTree("NodeX")
    right = ExprTree("NodeY")
    return ExprTree("OpBinary", left, right)


@pytest.fixture
def commutative_tree():
    """Create a tree with commutative operation."""
    left = ExprTree("NodeX")
    right = ExprTree("NodeY")
    return ExprTree("OpCommutative", left, right, is_commutative=True)


class TestExprTree:
    """Test ExprTree for expression visualization and testing."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_leaf_node(self, leaf_node):
        """Test initialization of leaf node with no children."""
        assert leaf_node.node_type == "LeafA"
        assert leaf_node.children == ()
        assert leaf_node.is_commutative is False

    def test_init_with_children(self, simple_tree):
        """Test initialization with children."""
        assert simple_tree.node_type == "OpBinary"
        assert len(simple_tree.children) == 2
        assert simple_tree.children[0].node_type == "NodeX"
        assert simple_tree.children[1].node_type == "NodeY"

    def test_init_commutative_flag(self, commutative_tree):
        """Test initialization with is_commutative flag."""
        assert commutative_tree.is_commutative is True

    def test_init_multiple_children(self):
        """Test initialization with more than two children."""
        children = [ExprTree("ChildA"), ExprTree("ChildB"), ExprTree("ChildC")]
        tree = ExprTree("OpMulti", *children)
        assert len(tree.children) == 3
        assert all(isinstance(child, ExprTree) for child in tree.children)

    def test_init_default_is_commutative(self):
        """Test that is_commutative defaults to False."""
        tree = ExprTree("OpDefault", ExprTree("NodeX"), ExprTree("NodeY"))
        assert tree.is_commutative is False

    def test_init_single_child(self):
        """Test tree with only one child."""
        tree = ExprTree("OpUnary", ExprTree("LeafA"))
        assert len(tree.children) == 1
        assert tree.children[0].node_type == "LeafA"

    # ----------------------
    # Equality tests - non-commutative
    # ----------------------

    def test_eq_leaf_nodes_same_type(self):
        """Test equality of leaf nodes with same type."""
        tree1 = ExprTree("LeafA")
        tree2 = ExprTree("LeafA")
        assert tree1 == tree2

    def test_eq_leaf_nodes_different_type(self):
        """Test inequality of leaf nodes with different types."""
        tree1 = ExprTree("LeafA")
        tree2 = ExprTree("LeafB")
        assert tree1 != tree2

    def test_eq_non_commutative_same_order(self):
        """Test equality of non-commutative trees with same child order."""
        left1 = ExprTree("NodeX")
        right1 = ExprTree("NodeY")
        tree1 = ExprTree("OpSub", left1, right1)

        left2 = ExprTree("NodeX")
        right2 = ExprTree("NodeY")
        tree2 = ExprTree("OpSub", left2, right2)

        assert tree1 == tree2

    def test_eq_non_commutative_different_order(self):
        """Test inequality of non-commutative trees with different child order."""
        left1 = ExprTree("NodeX")
        right1 = ExprTree("NodeY")
        tree1 = ExprTree("OpSub", left1, right1)

        left2 = ExprTree("NodeY")
        right2 = ExprTree("NodeX")
        tree2 = ExprTree("OpSub", left2, right2)

        assert tree1 != tree2

    def test_eq_different_node_types(self):
        """Test inequality when node types differ."""
        tree1 = ExprTree("OpAdd", ExprTree("NodeX"), ExprTree("NodeY"))
        tree2 = ExprTree("OpMul", ExprTree("NodeX"), ExprTree("NodeY"))
        assert tree1 != tree2

    def test_eq_different_number_of_children(self):
        """Test inequality when number of children differs."""
        tree1 = ExprTree("Op", ExprTree("NodeX"))
        tree2 = ExprTree("Op", ExprTree("NodeX"), ExprTree("NodeY"))
        assert tree1 != tree2

    def test_eq_with_non_tree(self):
        """Test inequality when comparing with non-ExprTree object."""
        tree = ExprTree("LeafA")
        assert tree != "LeafA"
        assert tree != 42

    # ----------------------
    # Equality tests - commutative
    # ----------------------

    def test_eq_commutative_same_order(self):
        """Test equality of commutative trees with same child order."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        assert tree1 == tree2

    def test_eq_commutative_different_order(self):
        """Test equality of commutative trees with different child order.

        This is the key feature: commutative operations should be equal
        regardless of child order.
        """
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeY"), ExprTree("NodeX"), is_commutative=True
        )
        assert tree1 == tree2

    def test_eq_commutative_three_children_different_order(self):
        """Test equality of commutative trees with three children in different order."""
        tree1 = ExprTree(
            "OpAdd",
            ExprTree("NodeX"),
            ExprTree("NodeY"),
            ExprTree("NodeZ"),
            is_commutative=True,
        )
        tree2 = ExprTree(
            "OpAdd",
            ExprTree("NodeZ"),
            ExprTree("NodeX"),
            ExprTree("NodeY"),
            is_commutative=True,
        )
        tree3 = ExprTree(
            "OpAdd",
            ExprTree("NodeY"),
            ExprTree("NodeZ"),
            ExprTree("NodeX"),
            is_commutative=True,
        )
        assert tree1 == tree2
        assert tree2 == tree3
        assert tree1 == tree3

    def test_eq_commutative_flag_mismatch(self):
        """Test inequality when is_commutative flags differ.

        Even with same children, trees with different commutativity should not be equal.
        """
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=False
        )
        assert tree1 != tree2

    def test_eq_nested_commutative_operations(self):
        """Test equality with nested commutative operations."""
        # (NodeX + NodeY) + NodeZ vs (NodeY + NodeX) + NodeZ
        inner1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree1 = ExprTree("OpAdd", inner1, ExprTree("NodeZ"), is_commutative=True)

        inner2 = ExprTree(
            "OpAdd", ExprTree("NodeY"), ExprTree("NodeX"), is_commutative=True
        )
        tree2 = ExprTree("OpAdd", inner2, ExprTree("NodeZ"), is_commutative=True)

        assert tree1 == tree2

    # ----------------------
    # Hash tests
    # ----------------------

    def test_hash_equality_contract_leaf_nodes(self):
        """Test hash/equality contract: equal leaf nodes must have equal hashes."""
        tree1 = ExprTree("LeafA")
        tree2 = ExprTree("LeafA")
        assert hash(tree1) == hash(tree2)

    def test_hash_equality_contract_non_commutative(self):
        """Test hash/equality contract for non-commutative trees."""
        tree1 = ExprTree("OpSub", ExprTree("NodeX"), ExprTree("NodeY"))
        tree2 = ExprTree("OpSub", ExprTree("NodeX"), ExprTree("NodeY"))
        assert hash(tree1) == hash(tree2)

    def test_hash_equality_contract_commutative_same_order(self):
        """Test hash/equality contract for commutative trees with same order."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        assert hash(tree1) == hash(tree2)

    def test_hash_equality_contract_commutative_different_order(self):
        """Test hash/equality contract: commutative trees with different order."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeY"), ExprTree("NodeX"), is_commutative=True
        )
        assert hash(tree1) == hash(tree2)

    def test_hash_equality_contract_commutative_with_duplicates(self):
        """Test hash/equality contract with duplicate children."""
        tree1 = ExprTree(
            "OpAdd",
            ExprTree("NodeX"),
            ExprTree("NodeX"),
            ExprTree("NodeY"),
            is_commutative=True,
        )
        tree2 = ExprTree(
            "OpAdd",
            ExprTree("NodeY"),
            ExprTree("NodeX"),
            ExprTree("NodeX"),
            is_commutative=True,
        )
        assert tree1 == tree2
        assert hash(tree1) == hash(tree2)

    def test_hash_stability(self):
        """Test that the same object always produces the same hash."""
        tree = ExprTree("OpAdd", ExprTree("NodeX"), ExprTree("NodeY"))
        hash1 = hash(tree)
        hash2 = hash(tree)
        assert hash1 == hash2

    def test_hash_commutative_flag_affects_hash(self):
        """Test that is_commutative flag affects the hash value."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=False
        )

        assert hash(tree1) != hash(tree2)

    def test_hash_different_for_different_counts(self):
        """Test that different duplicate counts produce different hashes."""
        tree1 = ExprTree(
            "OpAdd",
            ExprTree("NodeX"),
            ExprTree("NodeX"),
            ExprTree("NodeY"),
            is_commutative=True,
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        assert hash(tree1) != hash(tree2)

    def test_hash_in_set(self):
        """Test that equal trees are treated as one element in a set."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeY"), ExprTree("NodeX"), is_commutative=True
        )
        tree3 = ExprTree("OpMul", ExprTree("NodeX"), ExprTree("NodeY"))

        tree_set = {tree1, tree2, tree3}

        assert len(tree_set) == 2

    def test_hash_in_dict(self):
        """Test that equal trees map to the same dictionary key."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        tree2 = ExprTree(
            "OpAdd", ExprTree("NodeY"), ExprTree("NodeX"), is_commutative=True
        )

        tree_dict = {tree1: "first"}
        tree_dict[tree2] = "second"

        # tree1 and tree2 are equal, so should map to same key
        assert len(tree_dict) == 1
        assert tree_dict[tree1] == "second"
        assert tree_dict[tree2] == "second"

    # ----------------------
    # Repr tests
    # ----------------------

    def test_repr_leaf_node(self):
        """Test __repr__ for leaf node without children."""
        tree = ExprTree("LeafA")
        assert repr(tree) == "ExprTree('LeafA')"

    def test_repr_leaf_node_commutative(self):
        """Test __repr__ for leaf node with is_commutative=True."""
        tree = ExprTree("LeafA", is_commutative=True)
        assert repr(tree) == "ExprTree('LeafA', is_commutative=True)"

    def test_repr_simple_tree(self):
        """Test __repr__ for tree with children."""
        tree = ExprTree("OpAdd", ExprTree("NodeX"), ExprTree("NodeY"))
        expected = "ExprTree('OpAdd', ExprTree('NodeX'), ExprTree('NodeY'))"
        assert repr(tree) == expected

    def test_repr_simple_tree_commutative(self):
        """Test __repr__ for commutative tree shows is_commutative flag."""
        tree = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        expected = (
            "ExprTree('OpAdd', ExprTree('NodeX'), ExprTree('NodeY'), "
            "is_commutative=True)"
        )
        assert repr(tree) == expected

    def test_repr_nested_tree(self):
        """Test __repr__ for nested tree structure."""
        inner = ExprTree("OpMul", ExprTree("NodeX"), ExprTree("NodeY"))
        outer = ExprTree("OpAdd", inner, ExprTree("NodeZ"))
        expected = (
            "ExprTree('OpAdd', "
            "ExprTree('OpMul', ExprTree('NodeX'), ExprTree('NodeY')), "
            "ExprTree('NodeZ'))"
        )
        assert repr(outer) == expected

    def test_repr_three_children(self):
        """Test __repr__ for tree with three children."""
        tree = ExprTree(
            "OpMulti", ExprTree("NodeX"), ExprTree("NodeY"), ExprTree("NodeZ")
        )
        expected = (
            "ExprTree('OpMulti', ExprTree('NodeX'), ExprTree('NodeY'), "
            "ExprTree('NodeZ'))"
        )
        assert repr(tree) == expected

    def test_repr_nested_commutative(self):
        """Test __repr__ with nested commutative operations."""
        inner = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        outer = ExprTree("OpMul", inner, ExprTree("NodeZ"), is_commutative=True)
        expected = (
            "ExprTree('OpMul', "
            "ExprTree('OpAdd', ExprTree('NodeX'), ExprTree('NodeY'), "
            "is_commutative=True), "
            "ExprTree('NodeZ'), "
            "is_commutative=True)"
        )
        assert repr(outer) == expected

    def test_repr_evaluable(self):
        """Test that repr output can be evaluated to recreate equivalent tree."""
        tree1 = ExprTree(
            "OpAdd", ExprTree("NodeX"), ExprTree("NodeY"), is_commutative=True
        )
        # Get repr and evaluate it
        tree_repr = repr(tree1)
        assert tree1 == eval(tree_repr)

    # ----------------------
    # String tests
    # ----------------------

    def test_str_leaf_node(self):
        """Test __str__ for leaf node."""
        tree = ExprTree("LeafA")
        result = str(tree)
        assert result == "LeafA\n"

    def test_str_single_child(self):
        """Test __str__ for tree with single child."""
        tree = ExprTree("OpUnary", ExprTree("NodeX"))
        result = str(tree)

        # fmt: off
        expected = (
            "OpUnary\n"
            "└─ NodeX\n"
        )
        # fmt: on

        assert result == expected

    def test_str_two_children(self):
        """Test __str__ for simple tree with two children."""
        tree = ExprTree("OpAdd", ExprTree("NodeX"), ExprTree("NodeY"))
        result = str(tree)

        # fmt: off
        expected = (
            "OpAdd\n"
            "├─ NodeX\n"
            "└─ NodeY\n"
        )
        # fmt: on

        assert result == expected

    def test_str_three_children(self):
        """Test __str__ for tree with three children."""
        tree = ExprTree(
            "OpMulti", ExprTree("NodeX"), ExprTree("NodeY"), ExprTree("NodeZ")
        )
        result = str(tree)

        # fmt: off
        expected = (
            "OpMulti\n"
            "├─ NodeX\n"
            "├─ NodeY\n"
            "└─ NodeZ\n"
        )
        # fmt: on

        assert result == expected

    def test_str_nested_two_levels(self):
        """Test __str__ for nested tree structure (two levels deep)."""
        inner = ExprTree("OpMul", ExprTree("NodeX"), ExprTree("NodeY"))
        outer = ExprTree("OpAdd", inner, ExprTree("NodeZ"))
        result = str(outer)

        # fmt: off
        expected = (
            "OpAdd\n"
            "├─ OpMul\n"
            "│  ├─ NodeX\n"
            "│  └─ NodeY\n"
            "└─ NodeZ\n"
        )
        # fmt: on

        assert result == expected

    def test_str_deeply_nested(self):
        """Test __str__ for deeply nested tree (three levels)."""
        innermost = ExprTree("Level3", ExprTree("LeafA"), ExprTree("LeafB"))
        middle = ExprTree("Level2", innermost, ExprTree("LeafC"))
        outer = ExprTree("Level1", middle, ExprTree("LeafD"))
        result = str(outer)

        # fmt: off
        expected = (
            "Level1\n"
            "├─ Level2\n"
            "│  ├─ Level3\n"
            "│  │  ├─ LeafA\n"
            "│  │  └─ LeafB\n"
            "│  └─ LeafC\n"
            "└─ LeafD\n"
        )
        # fmt: on

        assert result == expected

    def test_str_multiple_nested_children(self):
        """Test __str__ with multiple nested children."""
        left = ExprTree("OpLeft", ExprTree("L1"), ExprTree("L2"))
        right = ExprTree("OpRight", ExprTree("R1"), ExprTree("R2"))
        tree = ExprTree("OpRoot", left, right)
        result = str(tree)

        # fmt: off
        expected = (
            "OpRoot\n"
            "├─ OpLeft\n"
            "│  ├─ L1\n"
            "│  └─ L2\n"
            "└─ OpRight\n"
            "   ├─ R1\n"
            "   └─ R2\n"
        )
        # fmt: on

        assert result == expected
