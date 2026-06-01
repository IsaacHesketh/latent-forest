"""
Autograd implementation
"""

import math


class Value:
    """
    ``Value`` wraps a single scalar number ``data`` and tracks how it was computed.

    Every time you do math with Value objects (``add``, ``multiply``, etc.), the result is a new Value that remembers its
    inputs (``_children``) and the local derivative of that operation (``_local_grads``).
    """

    __slots__ = (
        "data",
        "grad",
        "_children",
        "_local_grads",
    )  # Using slots to reduce memory load

    def __init__(self, data, children=(), local_grads=()):
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = (
            0  # derivative of the loss w.r.t. this node, calculated in backward pass
        )
        self._children = children  # children of this node in the computation graph
        self._local_grads = (
            local_grads  # local derivative of this node w.r.t. its children
        )

    def __add__(self, other):
        """
        Add another nodes value to this one

        Args:
            other: Other node to add

        Returns:
            New ``Value`` object with sum of the inputs
        """
        # Confirm other is a Value
        if not isinstance(other, Value):
            other = Value(other)

        # Add the two nodes
        return Value(
            data=self.data + other.data,  # data is the sum of the two inputs
            children=(self, other),  # children
            local_grads=(1, 1),  # local derivative of this node w.r.t its children
        )

    def __mul__(self, other):
        """
        Multiply the value of this node with another

        Args:
            other: Other node to multiply

        Returns:
            New ``Value`` object with product of the inputs
        """
        # Confirm other is a Value
        if not isinstance(other, Value):
            other = Value(other)

        # Now multiply the two nodes
        return Value(
            data=self.data * other.data,  # Data is the product of the two inputs,
            children=(self, other),  # children,
            # TODO: look into why the gradients work like this
            local_grads=(
                other.data,
                self.data,
            ),  # local derivative of this node w.r.t its children
        )

    def __pow__(self, other):
        """
        Calculate the power of a vale

        Args:
            other: other: Other value for exponent

        Returns:
            New ``Value`` object with power of the inputs
        """
        # No need to check if other is a Value, other is a scalar
        return Value(
            data=self.data**other,  # Set this nodes data to the power of ``other``
            children=(self,),  # Only one child node,
            local_grads=(
                other * self.data ** (other - 1),
            ),  # Only one gradient, multiply by power and minus one
        )

    def log(self):
        """
        Calculate the log of a ``Value``

        Returns:
            New ``Value`` object that is the log of the input
        """
        # No need to check if other is a Value, log takes one input
        return Value(
            data=math.log(self.data),  # Calculate the log of the node
            children=(self,),  # Only one child
            local_grads=(1 / self.data,),
        )

    def exp(self):
        """
        Calculate exponential function

        Returns:
            New ``Value`` object that is the value of the exponential function
        """
        # No need to check if other is a Value
        return Value(
            data=math.exp(self.data),  # Exponential function e^x with x=self.data
            children=(self,),  # Only one child
            local_grads=(math.exp(self.data),),  # Differential of e^x is e^x
        )

    def relu(self):
        """
        Calculate rectified linear unit

        Returns:
            New ``Value`` object that is input directly if positive, otherwise returning zero.
        """
        # No need to check if other is a Value
        return Value(
            data=max(0, self.data),  # take input only if above zero
            children=(self,),  # Only one child
            local_grads=(float(self.data > 0),),  # 1 if data is pos, 0 otherwise
        )

    def __neg__(self):
        """
        Negative of input

        Returns:
            Negative ``Value``
        """
        return self * -1

    def __radd__(self, other):
        """

        Args:
            other: Other ``Value`` for addition

        Returns:

        """
        # TODO: research purpose of this
        return self + other

    def __sub__(self, other):
        """
        Subtract other from self

        Args:
            other: Other ``Value`` for subtraction

        Returns:
            New ``Value``
        """
        return self + (-other)

    def rsub(self, other):
        """
        Subtract self from other

        Args:
            other: Other ``Value`` for subtraction

        Returns:
            New ``Value``
        """
        return other + (-self)

    def __rmul__(self, other):
        """
        Multiply by other

        Args:
            other: Other ``Value`` for multiplication

        Returns:
            New ``Value``
        """
        return self * other

    def __truediv__(self, other):
        """
        Divide by other

        Args:
            other: Other ``Value`` for division

        Returns:
            New ``Value``
        """
        return self * (other**-1)

    def __rtruediv__(self, other):
        """
        Divide other by self
        Args:
            other: Other ``Value`` for division

        Returns:
            New ``Value``
        """
        return other * (self**-1)

    def backward(self):
        """
        Run backpropagation on a ``Value``
        """
        topo = []
        visited = set()

        def build_topo(v):
            # Check if the value has been visited already
            if v not in visited:
                # Add to visited set
                visited.add(v)

                # Loop over children, building topo recursively
                for child in v._children:
                    build_topo(child)

                # Add v to topo
                topo.append(v)

        # build topo on self
        build_topo(self)
        self.grad = 1

        # Loop over every value in topo, backwards from last added to first
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # update gradient of child
                child.grad += local_grad * v.grad
