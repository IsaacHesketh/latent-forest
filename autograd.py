"""
Autograd implementation
"""

class Value:
    """
    ``Value`` wraps a single scalar number ``data`` and tracks how it was computed.

    Every time you do math with Value objects (``add``, ``multiply``, etc.), the result is a new Value that remembers its
    inputs (``_children``) and the local derivative of that operation (``_local_grads``).
    """
    __slots__ = ("data", "grad", "_children", "_local_grads") # Using slots to reduce memory load

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children


    def add(self, other):
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
            data=self.data+other.data, # data is the sum of the two inputs
            children=(self, other), # children
            local_grads=(1,1)
        )