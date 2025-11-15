"""Tests for AtomExpression base class."""

import pytest
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import ExprTree, Variable


class MockAtom(AtomExpression):
    """Mock atom for testing AtomExpression base class."""

    def __init__(self, x, variable_only=False, buffer_value=None):
        """Initialize mock atom with configurable variable_only and optional buffer."""
        super().__init__(x, variable_only=variable_only, buffer=buffer_value)

    def is_smooth(self):
        """Always smooth for testing purposes."""
        return True

    def is_proxable(self):
        """Never proxable for testing purposes."""
        return False

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Mock prox method."""
        raise NotImplementedError("MockAtom does not implement prox.")

    def is_subsamplable(self):
        """Never subsamplable for testing purposes."""
        return False

    def subsample(self, indices: torch.Tensor):
        """Mock subsample method."""
        raise NotImplementedError("MockAtom does not implement subsample.")

    def forward(self):
        """Mock forward method."""
        return torch.tensor(0.0)


@pytest.fixture
def simple_variable():
    """Create a simple variable for testing."""
    return Variable((3,), name="x")


@pytest.fixture
def add_expression(simple_variable):
    """Create an Expression for testing."""
    return simple_variable + 1.0


class TestRegisterInputAndGetInput:
    """Tests for AtomExpression.register_input and get_input methods."""

    def test_register_input_with_variable_allows_variable(self, simple_variable):
        """Test register_input accepts Variable when variable_only=False."""
        atom = MockAtom(simple_variable, variable_only=False)
        assert atom.x1 is simple_variable

    def test_register_input_with_expression_when_allowed(self, add_expression):
        """Test register_input accepts Expression when variable_only=False."""
        atom = MockAtom(add_expression, variable_only=False)
        assert atom.x1 is add_expression

    def test_register_input_rejects_non_expression(self):
        """Test register_input rejects non-Expression inputs."""
        with pytest.raises(TypeError, match="Expected Expression, but got"):
            MockAtom(torch.tensor([1.0, 2.0, 3.0]), variable_only=False)

    def test_register_input_rejects_expression_when_variable_only(self, add_expression):
        """Test register_input rejects non-Variable when variable_only=True."""
        with pytest.raises(TypeError, match="Expected Variable, but got"):
            MockAtom(add_expression, variable_only=True)

    def test_register_input_accepts_variable_when_variable_only(self, simple_variable):
        """Test register_input accepts Variable when variable_only=True."""
        atom = MockAtom(simple_variable, variable_only=True)
        assert atom.x1 is simple_variable


class TestRegisterAtomBuffer:
    """Tests for AtomExpression.register_atom_buffer method."""

    def test_register_buffer_with_float(self, simple_variable):
        """Test register_atom_buffer converts float to tensor."""
        atom = MockAtom(simple_variable, buffer_value=2.5)
        assert isinstance(atom.buffer, torch.Tensor)
        assert torch.isclose(atom.buffer, torch.tensor(2.5))

    def test_register_buffer_with_tensor(self, simple_variable):
        """Test register_atom_buffer accepts tensor."""
        buffer_tensor = torch.tensor([1.0, 2.0, 3.0])
        atom = MockAtom(simple_variable, buffer_value=buffer_tensor)
        assert isinstance(atom.buffer, torch.Tensor)
        assert torch.equal(atom.buffer, buffer_tensor)

    def test_register_buffer_with_none(self, simple_variable):
        """Test register_atom_buffer accepts None."""
        atom = MockAtom(simple_variable, buffer_value=None)
        assert atom.buffer is None

    @pytest.mark.parametrize(
        "invalid_buffer",
        [
            "string",
            [1.0, 2.0],
            {"key": "value"},
            42,
        ],
        ids=["str", "list", "dict", "int"],
    )
    def test_register_buffer_rejects_invalid_types(
        self, simple_variable, invalid_buffer
    ):
        """Test register_atom_buffer rejects invalid buffer types."""
        with pytest.raises(TypeError, match="Expected float, Tensor, or None"):
            MockAtom(simple_variable, buffer_value=invalid_buffer)


class TestTree:
    """Tests for AtomExpression.tree() method."""

    def test_tree_with_variable_input(self, simple_variable):
        """Test tree() returns correct structure for atom with Variable input."""
        atom = MockAtom(simple_variable, variable_only=False)
        expected = ExprTree("MockAtom", ExprTree("Variable(x)"))
        assert atom.tree() == expected

    def test_tree_with_expression_input(self, add_expression):
        """Test tree() returns correct structure for atom with Expression input."""
        atom = MockAtom(add_expression, variable_only=False)
        expected = ExprTree(
            "MockAtom",
            ExprTree(
                "AddExpression",
                ExprTree("Variable(x)"),
                ExprTree("Constant"),
                is_commutative=True,
            ),
        )
        assert atom.tree() == expected

    def test_tree_structure_preserves_nesting(self, simple_variable):
        """Test tree() correctly represents nested atom structures."""
        inner_atom = MockAtom(simple_variable, variable_only=False)
        outer_atom = MockAtom(inner_atom, variable_only=False)
        expected = ExprTree("MockAtom", ExprTree("MockAtom", ExprTree("Variable(x)")))
        assert outer_atom.tree() == expected
