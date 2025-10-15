
# Retrograd
# A lightweight automatic differentiation engine.

class Value:
  def __init__(self, data, grad=0, _children=(), _op='', label=''):
    """Initialize the class with parameters.
    Args:
      data: data held in Value
      grad: cumulative gradient
      children: set of children Value nodes
      label: label of the Value node (for visualization)
    """
    self.data = data
    self.grad = grad
    self._prev = set(_children)
    self.label = label
    self._op = _op
  
  def __repr__(self):
    """String representation of the class."""
    return f"Value({self.label} | Data={self.data}, Grad={self.grad})"
  
  def __add__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data,_children=(self,other),_op='+')
    return out
  
  def __mul__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data,_children=(self,other),_op='*')
    return out
