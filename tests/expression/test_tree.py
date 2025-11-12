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
