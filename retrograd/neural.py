import random
from retrograd.engine import Value

class Neuron:
  def __init__(self,n_inputs):
    self.w = [Value(random.uniform(-1,1)) for _ in range (n_inputs)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self,x):
    # w * x + b  <-- w and x are vectors
    act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
    out = act.tanh()
    return out
  
  def get_parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self,n_inputs,n_outputs):
    self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

  def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs
  
  def get_parameters(self):
    return [p for neuron in self.neurons for p in neuron.get_parameters()]
  
class MLP:
  def __init__(self,n_inputs,layer_sizes):
    sizes = [n_inputs] + layer_sizes
    self.layers = [Layer(sizes[i],sizes[i+1]) for i in range(len(layer_sizes))]

  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def get_parameters(self):
    return [p for layer in self.layers for p in layer.get_parameters()]
  

