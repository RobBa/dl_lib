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