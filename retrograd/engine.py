
# Retrograd
# A lightweight automatic differentiation engine.

import math
class Value:
  def __init__(self, data, grad=0.0, _children=(), _op='', label=''):
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
    self._backward = lambda: None
    self.label = label
    self._op = _op
  
  def __repr__(self):
    """String representation of the class."""
    return f"Value({self.label} | Data={self.data}, Grad={self.grad})"
  
  def __add__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data,_children=(self,other),_op='+')
    def _backward():
      self.grad = 1.0 * out.grad
      other.grad = 1.0 * out.grad
    out._backward = _backward
    return out
  
  def __mul__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data,_children=(self,other),_op='*')
    def _backward():
      self.grad = other.data * out.grad
      other.grad = self.data * out.grad
    out._backward = _backward
    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t,_children=(self,), _op='tanh')
    def _backward():
      self.grad = (1.0 - t**2) * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    # topological order of the children graph
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad=1.0
    for node in reversed(topo):
      node._backward() #note this is the internal _backward function
          
    
