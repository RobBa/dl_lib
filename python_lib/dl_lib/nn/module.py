"""
Module base class. We use it to automatically register network 
modules when defining graphs via Module.
"""

from .._compiled._nn import _Module

class Module(_Module):
  def __init__(self):
    object.__setattr__(self, "_modules", {}) # not necessary, but more explicit
    self._modules = {} 
  
  """
  Stores attributes defined in __init__ in private 
  _modules dictionary
  """
  def __setattr__(self, name, value):
    if isinstance(value, Module):
      self._modules[name] = value
    object.__setattr__(self, name, value)
  
  """
  Returns a list of leaf parameters. Used to identify trainable
  nodes of a graph.
  """
  def parameters(self):
    params = self._own_parameters()  # calls C++ side for leaf modules
    for module in self._modules.values():
      params.extend(module.parameters())
    return params
  
"""
For convenience.
"""
class Sequential(Module):
  def __init__(self):
    super().__init__()
    object.__setattr__(self, "_layers", [])

  def append(self, module):
    self._layers.append(module)

  def forward(self, x):
    for layer in self._layers:
      x = layer(x)
    return x

  def parameters(self):
    params = []
    for layer in self._layers:
      if hasattr(layer, 'parameters'):
        result = layer.parameters()
        if isinstance(result, list):
          params.extend(result)
        else:
          params.extend(list(result))  # force conversion from BP proxy
      elif hasattr(layer, 'params'):
        params.extend(list(layer.params))
    return params