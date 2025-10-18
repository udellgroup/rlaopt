"""Comprehensive tests for the AtomExpression base class."""

import pytest
import torch

from rlaopt.atoms import AtomExpression, InputType
from rlaopt.expression import Expression, Variable

# ===============================
# Test InputType Enum
# ===============================


class TestInputType:
    """Tests for InputType enum."""

    def test_variable_value(self):
        """Test VARIABLE enum value."""
        assert InputType.VARIABLE.value == "variable"

    def test_expression_value(self):
        """Test EXPRESSION enum value."""
        assert InputType.EXPRESSION.value == "expression"

    def test_enum_members(self):
        """Test that enum has exactly two members."""
        assert len(InputType) == 2
        assert InputType.VARIABLE in InputType
        assert InputType.EXPRESSION in InputType

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert InputType.VARIABLE != InputType.EXPRESSION
        assert InputType.VARIABLE == InputType.VARIABLE


# ===============================
# Concrete Test Atom
# ===============================


class ConcreteAtom(AtomExpression):
    """Concrete implementation of AtomExpression for testing."""

    def __init__(self):
        """Initialize ConcreteAtom."""
        super().__init__()
        self._is_smooth = True
        self._is_proxable = False
        self._is_subsamplable = False

    def is_smooth(self) -> bool:
        """Check if atom is smooth."""
        return self._is_smooth

    def is_proxable(self) -> bool:
        """Check if atom is proxable."""
        return self._is_proxable

    def forward(self) -> torch.Tensor:
        """Compute forward pass."""
        return torch.tensor(0.0)

    def is_subsamplable(self) -> bool:
        """Check if atom is subsamplable."""
        return self._is_subsamplable

    def subsample(self, indices: torch.Tensor) -> AtomExpression:
        """Subsample the atom."""
        if not self._is_subsamplable:
            raise NotImplementedError("Atom is not subsamplable")
        # Return a new instance for testing
        return ConcreteAtom()

    def to_cvxpy(self):
        """Convert to CVXPY expression."""
        import cvxpy as cp

        return cp.Constant(0)


class SubsamplableAtom(AtomExpression):
    """Concrete subsamplable atom for testing."""

    def __init__(self, data: torch.Tensor = None):
        """Initialize SubsamplableAtom with optional data."""
        super().__init__()
        if data is not None:
            self.register_atom_buffer("data", data)
        self._subsample_indices = None

    def is_smooth(self) -> bool:
        """Check if atom is smooth."""
        return True

    def is_proxable(self) -> bool:
        """Check if atom is proxable."""
        return False

    def forward(self) -> torch.Tensor:
        """Compute forward pass."""
        if self._subsample_indices is not None:
            return self.data[self._subsample_indices].sum()
        return self.data.sum() if hasattr(self, "data") else torch.tensor(0.0)

    def is_subsamplable(self) -> bool:
        """Check if atom is subsamplable."""
        return True

    def subsample(self, indices: torch.Tensor) -> AtomExpression:
        """Subsample the atom."""
        new_atom = SubsamplableAtom(self.data)
        new_atom._subsample_indices = indices
        return new_atom

    def to_cvxpy(self):
        """Convert to CVXPY expression."""
        import cvxpy as cp

        return cp.Constant(0)


# ===============================
# Test AtomExpression Base Class
# ===============================


class TestAtomExpressionInit:
    """Tests for AtomExpression initialization."""

    def test_initialization(self):
        """Test that concrete atom can be initialized."""
        atom = ConcreteAtom()
        assert isinstance(atom, AtomExpression)
        assert isinstance(atom, Expression)

    def test_is_expression_subclass(self):
        """Test that AtomExpression is a subclass of Expression."""
        assert issubclass(AtomExpression, Expression)

    def test_is_torch_module(self):
        """Test that AtomExpression is a torch.nn.Module."""
        atom = ConcreteAtom()
        assert isinstance(atom, torch.nn.Module)

    def test_cannot_instantiate_abstract_class(self):
        """Test that AtomExpression cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AtomExpression()


# ===============================
# Test Abstract Methods
# ===============================


class TestAbstractMethods:
    """Tests for abstract method enforcement."""

    def test_is_subsamplable_is_abstract(self):
        """Test that is_subsamplable must be implemented."""

        class IncompleteAtom(AtomExpression):
            """Incomplete atom missing is_subsamplable."""

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                return torch.tensor(0.0)

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

            # Missing: is_subsamplable, subsample

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAtom()

    def test_subsample_is_abstract(self):
        """Test that subsample must be implemented."""

        class IncompleteAtom(AtomExpression):
            """Incomplete atom missing subsample."""

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                return torch.tensor(0.0)

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

            # Missing: subsample

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAtom()


# ===============================
# Test register_variable
# ===============================


class TestRegisterVariable:
    """Tests for register_variable method."""

    def test_register_variable(self):
        """Test registering a variable."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")

        atom.register_variable(x)

        assert hasattr(atom, "var_name")
        assert atom.var_name == "x"
        assert hasattr(atom, "x")

    def test_registered_variable_is_parameter(self):
        """Test that registered variable becomes a parameter."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 3

        atom.register_variable(x)

        # Should be registered as a parameter
        params = dict(atom.named_parameters())
        assert "x" in params
        assert torch.allclose(params["x"], torch.ones(5) * 3)

    def test_register_multiple_variables(self):
        """Test registering multiple variables."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((3,), name="y")

        atom.register_variable(x)
        # Note: register_variable sets self.var_name, so only last one is stored
        # This is actually a limitation of the current implementation
        atom.var_name_x = atom.var_name  # Save first
        atom.register_variable(y)

        assert atom.var_name == "y"
        assert hasattr(atom, "x")
        assert hasattr(atom, "y")

    def test_get_variable_after_registration(self):
        """Test retrieving variable after registration."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 7

        atom.register_variable(x)
        retrieved = atom.get_variable("x")

        assert torch.allclose(retrieved, torch.ones(5) * 7)

    def test_variable_gradient_tracking(self):
        """Test that registered variables track gradients."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x", requires_grad=True)

        atom.register_variable(x)

        assert atom.get_variable("x").requires_grad is True


# ===============================
# Test register_expression
# ===============================


class TestRegisterExpression:
    """Tests for register_expression method."""

    def test_register_expression(self):
        """Test registering an expression."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y

        atom.register_expression(expr)

        assert hasattr(atom, "expr_name")
        assert atom.expr_name == "AddExpression"

    def test_registered_expression_is_submodule(self):
        """Test that registered expression becomes a submodule."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y

        atom.register_expression(expr)

        # Should be registered as a submodule
        submodules = dict(atom.named_modules())
        assert "AddExpression" in submodules

    def test_expression_parameters_tracked(self):
        """Test that expression parameters are tracked through atom."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y

        atom.register_expression(expr)

        # Atom should track expression's parameters
        params = list(atom.parameters())
        assert len(params) == 2  # x and y

    def test_register_nested_expression(self):
        """Test registering a nested expression."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        z = Variable((5,), name="z")
        expr = (x + y) * z

        atom.register_expression(expr)

        assert hasattr(atom, "expr_name")
        # Should track all three parameters
        params = list(atom.parameters())
        assert len(params) == 3


# ===============================
# Test register_input
# ===============================


class TestRegisterInput:
    """Tests for register_input method."""

    def test_register_input_with_variable(self):
        """Test register_input with a Variable."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")

        atom.register_input(x)

        assert atom.input_type == InputType.VARIABLE
        assert hasattr(atom, "var_name")
        assert atom.var_name == "x"

    def test_register_input_with_expression(self):
        """Test register_input with an Expression."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y

        atom.register_input(expr)

        assert atom.input_type == InputType.EXPRESSION
        assert hasattr(atom, "expr_name")


# ===============================
# Test register_atom_buffer
# ===============================


class TestRegisterAtomBuffer:
    """Tests for register_atom_buffer method."""

    def test_register_buffer_with_float(self):
        """Test registering a buffer with float value."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("scaling", 2.5)

        assert hasattr(atom, "scaling")
        assert torch.allclose(atom.scaling, torch.tensor(2.5))

    def test_register_buffer_with_int(self):
        """Test registering a buffer with int value."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("count", 10)

        assert hasattr(atom, "count")
        assert torch.allclose(atom.count, torch.tensor(10.0))

    def test_register_buffer_with_tensor(self):
        """Test registering a buffer with tensor value."""
        atom = ConcreteAtom()
        data = torch.tensor([1.0, 2.0, 3.0])
        atom.register_atom_buffer("data", data)

        assert hasattr(atom, "data")
        assert torch.equal(atom.data, data)

    def test_register_buffer_with_parameter(self):
        """Test registering a buffer with Parameter value."""
        atom = ConcreteAtom()
        param = torch.nn.Parameter(torch.tensor([4.0, 5.0, 6.0]))
        atom.register_atom_buffer("weights", param)

        assert hasattr(atom, "weights")
        assert torch.equal(atom.weights, param.data)

    def test_buffer_is_not_parameter(self):
        """Test that buffers are not registered as parameters."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("scaling", 2.5)

        # Should not be in parameters
        params = dict(atom.named_parameters())
        assert "scaling" not in params

        # Should be in buffers
        buffers = dict(atom.named_buffers())
        assert "scaling" in buffers

    def test_multiple_buffers(self):
        """Test registering multiple buffers."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("alpha", 1.0)
        atom.register_atom_buffer("beta", 2.0)
        atom.register_atom_buffer("gamma", torch.tensor([3.0, 4.0]))

        assert hasattr(atom, "alpha")
        assert hasattr(atom, "beta")
        assert hasattr(atom, "gamma")

        buffers = dict(atom.named_buffers())
        assert len(buffers) == 3

    def test_buffer_in_state_dict(self):
        """Test that buffers appear in state_dict."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("scaling", 2.5)

        state = atom.state_dict()
        assert "scaling" in state
        assert torch.allclose(state["scaling"], torch.tensor(2.5))


# ===============================
# Test get_variable
# ===============================


class TestGetVariable:
    """Tests for get_variable method."""

    def test_get_variable(self):
        """Test getting a registered variable."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 5

        atom.register_variable(x)
        result = atom.get_variable("x")

        assert torch.allclose(result, torch.ones(5) * 5)

    def test_get_nonexistent_variable(self):
        """Test getting a non-existent variable raises AttributeError."""
        atom = ConcreteAtom()

        with pytest.raises(AttributeError):
            atom.get_variable("nonexistent")

    def test_get_variable_returns_parameter(self):
        """Test that get_variable returns a Parameter."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")

        atom.register_variable(x)
        result = atom.get_variable("x")

        assert isinstance(result, torch.nn.Parameter)

    def test_get_variable_different_shapes(self):
        """Test getting variables with different shapes."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((3, 4), name="y")

        atom.register_variable(x)
        # Store var_name for x
        x_name = atom.var_name
        atom.register_variable(y)

        result_x = atom.get_variable(x_name)
        result_y = atom.get_variable("y")

        assert result_x.shape == torch.Size([5])
        assert result_y.shape == torch.Size([3, 4])


# ===============================
# Test expr_type
# ===============================


class TestExprType:
    """Tests for expr_type static method."""

    def test_expr_type_with_variable(self):
        """Test expr_type returns VARIABLE for Variable."""
        x = Variable((5,), name="x")
        result = AtomExpression.expr_type(x)
        assert result == InputType.VARIABLE

    def test_expr_type_with_expression(self):
        """Test expr_type returns EXPRESSION for Expression."""
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y
        result = AtomExpression.expr_type(expr)
        assert result == InputType.EXPRESSION

    def test_expr_type_with_complex_expression(self):
        """Test expr_type with complex expression."""
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        z = Variable((5,), name="z")
        expr = (x + y) * z
        result = AtomExpression.expr_type(expr)
        assert result == InputType.EXPRESSION

    def test_expr_type_is_static(self):
        """Test that expr_type can be called without instance."""
        x = Variable((5,), name="x")
        # Should work without creating an atom instance
        result = AtomExpression.expr_type(x)
        assert result == InputType.VARIABLE


# ===============================
# Test is_subsamplable and subsample
# ===============================


class TestSubsampling:
    """Tests for subsampling functionality."""

    def test_is_subsamplable_false(self):
        """Test atom that is not subsamplable."""
        atom = ConcreteAtom()
        assert atom.is_subsamplable() is False

    def test_is_subsamplable_true(self):
        """Test atom that is subsamplable."""
        data = torch.arange(10, dtype=torch.float32)
        atom = SubsamplableAtom(data)
        assert atom.is_subsamplable() is True

    def test_subsample_raises_when_not_subsamplable(self):
        """Test subsample raises error when atom is not subsamplable."""
        atom = ConcreteAtom()
        indices = torch.tensor([0, 1, 2])

        with pytest.raises(NotImplementedError, match="not subsamplable"):
            atom.subsample(indices)

    def test_subsample_returns_atom(self):
        """Test subsample returns an AtomExpression."""
        data = torch.arange(10, dtype=torch.float32)
        atom = SubsamplableAtom(data)
        indices = torch.tensor([0, 2, 4])

        subsampled = atom.subsample(indices)

        assert isinstance(subsampled, AtomExpression)
        assert isinstance(subsampled, SubsamplableAtom)

    def test_subsample_with_indices(self):
        """Test subsample creates correct subset."""
        data = torch.arange(10, dtype=torch.float32)
        atom = SubsamplableAtom(data)
        indices = torch.tensor([1, 3, 5])

        subsampled = atom.subsample(indices)

        # The subsampled atom should only use selected indices
        result = subsampled.forward()
        expected = data[indices].sum()
        assert torch.allclose(result, expected)

    def test_subsample_empty_indices(self):
        """Test subsample with empty indices."""
        data = torch.arange(10, dtype=torch.float32)
        atom = SubsamplableAtom(data)
        indices = torch.tensor([], dtype=torch.long)

        subsampled = atom.subsample(indices)
        result = subsampled.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_subsample_single_index(self):
        """Test subsample with single index."""
        data = torch.arange(10, dtype=torch.float32)
        atom = SubsamplableAtom(data)
        indices = torch.tensor([5])

        subsampled = atom.subsample(indices)
        result = subsampled.forward()

        assert torch.allclose(result, torch.tensor(5.0))


# ===============================
# Test Integration Scenarios
# ===============================


class TestAtomIntegration:
    """Integration tests for AtomExpression."""

    def test_atom_with_variable_forward(self):
        """Test atom forward pass with registered variable."""

        class SimpleAtom(AtomExpression):
            """Simple test atom."""

            def __init__(self, x: Variable):
                """Initialize with variable."""
                super().__init__()
                self.register_variable(x)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                return self.get_variable(self.var_name).sum()

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 3
        atom = SimpleAtom(x)

        result = atom.forward()
        assert torch.allclose(result, torch.tensor(15.0))

    def test_atom_with_expression_forward(self):
        """Test atom forward pass with registered expression."""

        class ExprAtom(AtomExpression):
            """Expression-based test atom."""

            def __init__(self, expr: Expression):
                """Initialize with expression."""
                super().__init__()
                self.register_expression(expr)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                expr_module = getattr(self, self.expr_name)
                return expr_module.forward().sum()

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        x.value.data = torch.ones(5) * 2
        y.value.data = torch.ones(5) * 3
        expr = x + y

        atom = ExprAtom(expr)
        result = atom.forward()

        # (2 + 3) * 5 = 25
        assert torch.allclose(result, torch.tensor(25.0))

    def test_atom_with_buffer_and_variable(self):
        """Test atom using both buffer and variable."""

        class ScaledAtom(AtomExpression):
            """Scaled test atom."""

            def __init__(self, x: Variable, scaling: float):
                """Initialize with variable and scaling."""
                super().__init__()
                self.register_variable(x)
                self.register_atom_buffer("scaling", scaling)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                return self.scaling * self.get_variable(self.var_name).sum()

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 4
        atom = ScaledAtom(x, scaling=2.5)

        result = atom.forward()
        # 2.5 * (4 * 5) = 50
        assert torch.allclose(result, torch.tensor(50.0))

    def test_atom_gradient_flow(self):
        """Test that gradients flow through atom."""

        class SumSquaresAtom(AtomExpression):
            """Sum squares test atom."""

            def __init__(self, x: Variable):
                """Initialize with variable."""
                super().__init__()
                self.register_variable(x)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                val = self.get_variable(self.var_name)
                return (val**2).sum()

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])
        atom = SumSquaresAtom(x)

        result = atom.forward()
        result.backward()

        # Gradient of x^2 is 2x
        expected_grad = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(x.value.grad, expected_grad)

    def test_atom_in_state_dict(self):
        """Test that atom's state can be saved and loaded."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 7

        atom = ConcreteAtom()
        atom.register_variable(x)
        atom.register_atom_buffer("scaling", 3.5)

        state = atom.state_dict()

        # Should contain both parameter and buffer
        assert "x" in state
        assert "scaling" in state
        assert torch.allclose(state["x"], torch.ones(5) * 7)
        assert torch.allclose(state["scaling"], torch.tensor(3.5))

    def test_atom_load_state_dict(self):
        """Test loading state into atom."""
        x = Variable((5,), name="x")
        atom = ConcreteAtom()
        atom.register_variable(x)

        # Create state with new values
        new_state = {"x": torch.ones(5) * 10}
        atom.load_state_dict(new_state, strict=False)

        # Values should be updated
        assert torch.allclose(atom.get_variable("x"), torch.ones(5) * 10)


# ===============================
# Test Edge Cases
# ===============================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_register_variable_overwrites_var_name(self):
        """Test that registering a new variable overwrites var_name."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((3,), name="y")

        atom.register_variable(x)
        assert atom.var_name == "x"

        atom.register_variable(y)
        assert atom.var_name == "y"  # Overwrites!

    def test_register_expression_overwrites_module_name(self):
        """Test that registering a new expression overwrites module_name."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr1 = x + y
        expr2 = x * y

        atom.register_expression(expr1)
        assert atom.expr_name == "AddExpression"

        atom.register_expression(expr2)
        assert atom.expr_name == "ProductExpression"  # Overwrites!

    def test_buffer_with_zero_value(self):
        """Test registering buffer with zero value."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("zero", 0.0)

        assert hasattr(atom, "zero")
        assert torch.allclose(atom.zero, torch.tensor(0.0))

    def test_buffer_with_negative_value(self):
        """Test registering buffer with negative value."""
        atom = ConcreteAtom()
        atom.register_atom_buffer("negative", -5.5)

        assert torch.allclose(atom.negative, torch.tensor(-5.5))

    def test_empty_atom(self):
        """Test creating an atom without registering anything."""
        atom = ConcreteAtom()

        # Should still be a valid module
        assert isinstance(atom, torch.nn.Module)
        assert len(list(atom.parameters())) == 0
        assert len(list(atom.buffers())) == 0

    def test_atom_to_device(self):
        """Test moving atom to device."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        atom.register_variable(x)
        atom.register_atom_buffer("scaling", 2.0)

        # Move to CPU (already there, but tests the method)
        atom = atom.to("cpu")

        assert atom.get_variable("x").device.type == "cpu"
        assert atom.scaling.device.type == "cpu"

    def test_atom_dtype_conversion(self):
        """Test converting atom dtype."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x", dtype=torch.float32)
        atom.register_variable(x)

        atom = atom.to(torch.float64)

        assert atom.get_variable("x").dtype == torch.float64

    def test_multiple_atoms_independent(self):
        """Test that multiple atom instances are independent."""
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")

        atom1 = ConcreteAtom()
        atom1.register_variable(x)

        atom2 = ConcreteAtom()
        atom2.register_variable(y)

        # Each should have its own var_name
        assert atom1.var_name == "x"
        assert atom2.var_name == "y"

        # Parameters should be independent
        assert "x" in dict(atom1.named_parameters())
        assert "y" in dict(atom2.named_parameters())
        assert "x" not in dict(atom2.named_parameters())
        assert "y" not in dict(atom1.named_parameters())


# ===============================
# Test with Mocks
# ===============================


class TestAtomWithMocks:
    """Tests using mocks to verify behavior."""

    def test_register_input_calls_register_variable(self):
        """Test that register_input calls register_variable for Variable."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")

        # Patch register_variable to verify it's called
        from unittest.mock import patch

        with patch.object(atom, "register_variable") as mock_register:
            atom.register_input(x)
            mock_register.assert_called_once_with(x)

    def test_register_input_calls_register_expression(self):
        """Test that register_input calls register_expression for Expression."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y

        from unittest.mock import patch

        with patch.object(atom, "register_expression") as mock_register:
            atom.register_input(expr)
            mock_register.assert_called_once_with(expr)

    def test_get_variable_retrieves_correct_attribute(self):
        """Test that get_variable retrieves the correct attribute."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((3,), name="y")

        atom.register_variable(x)
        x_name = atom.var_name  # Save x's name
        atom.register_variable(y)

        # Should retrieve the correct attributes
        x_val = atom.get_variable(x_name)
        y_val = atom.get_variable("y")

        assert x_val.shape == torch.Size([5])
        assert y_val.shape == torch.Size([3])

        # Should be the actual attributes
        assert x_val is getattr(atom, x_name)
        assert y_val is getattr(atom, "y")

    def test_register_atom_buffer_calls_register_buffer(self):
        """Test that register_atom_buffer calls register_buffer."""
        atom = ConcreteAtom()

        from unittest.mock import patch

        with patch.object(atom, "register_buffer") as mock_register:
            atom.register_atom_buffer("test", 5.0)
            mock_register.assert_called_once()

            # Check that it was called with the right arguments
            call_args = mock_register.call_args
            assert call_args[0][0] == "test"
            assert torch.allclose(call_args[0][1], torch.tensor(5.0))


# ===============================
# Test Concrete Use Cases
# ===============================


class TestConcreteUseCases:
    """Tests for realistic use cases."""

    def test_l1_norm_style_atom(self):
        """Test an L1-norm style atom."""

        class L1Norm(AtomExpression):
            """L1 norm atom."""

            def __init__(self, x: Variable, scaling: float = 1.0):
                """Initialize L1 norm."""
                super().__init__()
                self.register_variable(x)
                self.register_atom_buffer("scaling", scaling)

            def is_smooth(self):
                """Check if smooth."""
                return False

            def is_proxable(self):
                """Check if proxable."""
                return True

            def forward(self):
                """Compute forward."""
                val = self.get_variable(self.var_name)
                return self.scaling * torch.sum(torch.abs(val))

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError("L1Norm is not subsamplable")

            def to_cvxpy(self):
                """Convert to CVXPY."""
                import cvxpy as cp

                return cp.norm1(cp.Variable(5))

        x = Variable((5,), name="x")
        x.value.data = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])

        l1 = L1Norm(x, scaling=2.0)
        result = l1.forward()

        # 2.0 * (1 + 2 + 3 + 4 + 5) = 30
        assert torch.allclose(result, torch.tensor(30.0))

    def test_squared_loss_style_atom(self):
        """Test a squared loss style atom."""

        class SquaredLoss(AtomExpression):
            """Squared loss atom."""

            def __init__(self, x: Variable, target: torch.Tensor):
                """Initialize squared loss."""
                super().__init__()
                self.register_variable(x)
                self.register_atom_buffer("target", target)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                val = self.get_variable(self.var_name)
                diff = val - self.target
                return 0.5 * torch.sum(diff**2)

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])

        loss = SquaredLoss(x, target)
        result = loss.forward()

        # 0.5 * (0.5^2 + 0.5^2 + 0.5^2) = 0.5 * 0.75 = 0.375
        assert torch.allclose(result, torch.tensor(0.375))

    def test_linear_regression_style_atom(self):
        """Test a linear regression style atom with data buffer."""

        class LinearRegression(AtomExpression):
            """Linear regression atom."""

            def __init__(self, X: torch.Tensor, y: torch.Tensor, beta: Variable):
                """Initialize linear regression."""
                super().__init__()
                self.register_variable(beta)
                self.register_atom_buffer("X", X)
                self.register_atom_buffer("y", y)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                beta_val = self.get_variable(self.var_name)
                predictions = self.X @ beta_val
                residuals = predictions - self.y
                return 0.5 * torch.sum(residuals**2)

            def is_subsamplable(self):
                """Check if subsamplable."""
                return True

            def subsample(self, indices):
                """Subsample the atom."""
                new_atom = LinearRegression(
                    self.X[indices],
                    self.y[indices],
                    Variable(self.get_variable(self.var_name).data, name=self.var_name),
                )
                return new_atom

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        # Simple 2D linear regression
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        beta = Variable((2,), name="beta")
        beta.value.data = torch.tensor([0.5, 0.25])

        lr = LinearRegression(X, y, beta)
        result = lr.forward()

        # predictions = [1*0.5 + 2*0.25, 3*0.5 + 4*0.25, 5*0.5 + 6*0.25]
        #             = [1.0, 2.5, 4.0]
        # residuals = [0, 0.5, 1.0]
        # loss = 0.5 * (0 + 0.25 + 1.0) = 0.625
        assert torch.allclose(result, torch.tensor(0.625))

    def test_composite_atom_with_expression(self):
        """Test atom that takes an expression as input."""

        class NormOfExpression(AtomExpression):
            """Norm of expression atom."""

            def __init__(self, expr: Expression):
                """Initialize with expression."""
                super().__init__()
                self.register_expression(expr)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                expr_module = getattr(self, self.expr_name)
                val = expr_module.forward()
                return torch.norm(val)

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((3,), name="x")
        y = Variable((3,), name="y")
        x.value.data = torch.tensor([3.0, 0.0, 4.0])
        y.value.data = torch.tensor([0.0, 5.0, 0.0])

        expr = x + y
        atom = NormOfExpression(expr)
        result = atom.forward()

        # x + y = [3, 5, 4], norm = sqrt(9 + 25 + 16) = sqrt(50)
        expected = torch.sqrt(torch.tensor(50.0))
        assert torch.allclose(result, expected)


# ===============================
# Test Documentation Examples
# ===============================


class TestDocumentationExamples:
    """Test examples from docstrings."""

    def test_register_variable_example(self):
        """Test example from register_variable docstring."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        atom.register_variable(x)
        assert atom.var_name == "x"

    def test_get_variable_example(self):
        """Test example from get_variable docstring."""
        x = Variable((5,), name="x")
        atom = ConcreteAtom()
        atom.register_variable(x)
        param = atom.get_variable("x")
        assert isinstance(param, torch.nn.Parameter)

    def test_register_atom_buffer_example(self):
        """Test example from register_atom_buffer docstring."""

        class ScaledNorm(AtomExpression):
            """Scaled norm atom."""

            def __init__(self, x: Variable, scaling: float):
                """Initialize with variable and scaling."""
                super().__init__()
                self.register_variable(x)
                self.register_atom_buffer("scaling", scaling)

            def is_smooth(self):
                """Check if smooth."""
                return True

            def is_proxable(self):
                """Check if proxable."""
                return False

            def forward(self):
                """Compute forward."""
                return torch.tensor(0.0)

            def is_subsamplable(self):
                """Check if subsamplable."""
                return False

            def subsample(self, indices):
                """Subsample the atom."""
                raise NotImplementedError()

            def to_cvxpy(self):
                """Convert to CVXPY."""
                pass

        x = Variable((5,), name="x")
        atom = ScaledNorm(x, scaling=2.5)
        assert hasattr(atom, "scaling")
        assert torch.allclose(atom.scaling, torch.tensor(2.5))

    def test_register_input_variable_example(self):
        """Test Variable example from register_input docstring."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        atom.register_input(x)
        assert atom.input_type == InputType.VARIABLE

    def test_register_input_expression_example(self):
        """Test Expression example from register_input docstring."""
        atom = ConcreteAtom()
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y
        atom.register_input(expr)
        assert atom.input_type == InputType.EXPRESSION

    def test_expr_type_variable_example(self):
        """Test Variable example from expr_type docstring."""
        x = Variable((5,), name="x")
        result = AtomExpression.expr_type(x)
        assert result == InputType.VARIABLE

    def test_expr_type_expression_example(self):
        """Test Expression example from expr_type docstring."""
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y
        result = AtomExpression.expr_type(expr)
        assert result == InputType.EXPRESSION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
