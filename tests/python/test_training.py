"""
Robert Baumgartner, r.baumgartner-1@tudelft.nl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python_lib"))
print(sys.path)

from dl_lib import Tensor
from dl_lib.nn import FfLayer, Sequential
from dl_lib.nn.activation import LeakyReLU
from dl_lib.train.loss import BCE, BceWithSigmoid, CrossEntropyWithSoftmax
from dl_lib.train.optim import SGD, RmsProp

from dl_lib.sys import setSeed
import pytest

setSeed(42)

def train(net, loss_fn, optim, x, y, epochs):
  for epoch in range(epochs):
    ypred = net.forward(x)
    loss = loss_fn(y, ypred)

    loss.backward()
    optim.step()
    optim.zeroGrad()

  return loss

def make_binary_net():
  net = Sequential()
  net.append(FfLayer(2, 4, True, True))
  net.append(LeakyReLU(0.01))
  net.append(FfLayer(4, 1, True, True))
  return net

def make_multiclass_net():
  net = Sequential()
  net.append(FfLayer(2, 8, True, True))
  net.append(LeakyReLU(0.01))
  net.append(FfLayer(8, 3, True, True))
  return net

def make_xor_data():
  x = Tensor([4, 2], [0.0, 0.0,
                      0.0, 1.0,
                      1.0, 0.0,
                      1.0, 1.0], False)
  y = Tensor([4, 1], [0.0,
                      1.0,
                      1.0,
                      0.0], False)
  return x, y

def make_multiclass_data():
  x = Tensor([6, 2], [1.0, 0.0,
                      1.0, 0.1,
                      0.0, 1.0,
                      0.1, 1.0,
                      0.5, 0.5,
                      0.4, 0.6], False)
  y = Tensor([6, 3], [1.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0,
                      0.0, 0.0, 1.0], False)
  return x, y

class TestOverfitBinary:
  def test_binary_sgd_overfits(self):
    x, y = make_xor_data()
    net = make_binary_net()
    loss_fn = BceWithSigmoid()
    optim = SGD(net.parameters(), 0.05)

    final_loss = train(net, loss_fn, optim, x, y, epochs=2000)

    assert final_loss.getitem(0) < 0.05, \
      f"SGD failed to overfit XOR, loss={final_loss.getitem(0)}"

  def test_binary_rmsprop_overfits(self):
    x, y = make_xor_data()
    net = make_binary_net()
    loss_fn = BceWithSigmoid()
    optim = RmsProp(net.parameters(), 0.0001, 0.95)

    final_loss = train(net, loss_fn, optim, x, y, epochs=5000)

    assert final_loss.getitem(0) < 0.05, \
      f"RmsProp failed to overfit XOR, loss={final_loss.getitem(0)}"

  def test_multiclass_rmsprop_overfits(self):
    x, y = make_multiclass_data()
    net = make_multiclass_net()
    loss_fn = CrossEntropyWithSoftmax()
    optim = RmsProp(net.parameters(), 0.0003, 0.95)

    final_loss = train(net, loss_fn, optim, x, y, epochs=10000)

    assert final_loss.getitem(0) < 0.05, \
        f"RmsProp failed to overfit multiclass, loss={final_loss.getitem(0)}"

  def test_loss_decreases(self):
    """Loss should be strictly lower after training than before"""
    x, y = make_xor_data()
    net = make_binary_net()
    loss_fn = BceWithSigmoid()

    optim = SGD(net.parameters(), 0.001)
    initial_pred = net.forward(x)
    initial_loss = loss_fn(y, initial_pred).getitem(0)
    train(net, loss_fn, optim, x, y, epochs=2000)

    final_pred = net.forward(x)
    final_loss = loss_fn(y, final_pred).getitem(0)

    assert final_loss < initial_loss, \
      f"Loss did not decrease: {initial_loss} -> {final_loss}"
  