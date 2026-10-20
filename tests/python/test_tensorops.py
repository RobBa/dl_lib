"""
Robert Baumgartner, r.baumgartner-1@tudelft.nl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python_lib"))

from dl_lib import Tensor, Device
import pytest

class TestTensorOps():
  def test_ones(self):
    t = Tensor.ones([2, 2])
    assert t.dims == [2, 2]

  def test_ctor(self):
    t = Tensor([2], [1.0, 2.0], Device.CPU, False)

    assert t.getitem(0) == 1.0
    assert t.getitem(1) == 2.0

    assert t.device == Device.CPU
    assert t.requiresGrad == False

  def test_multiplication(self):
    a = Tensor.ones([2, 2]) * 3
    b = Tensor.ones([2, 2]) * 0.5
    c = a * b

    assert c.dims == [2, 2]
    assert c.getitem([0, 0]) == 1.5
    assert c.getitem([0, 1]) == 1.5
    assert c.getitem([1, 0]) == 1.5
    assert c.getitem([1, 1]) == 1.5

  def test_concat_dim0(self):
    a = Tensor([2, 2], [1, 2, 3, 4])
    b = Tensor([2, 2], [5, 6, 7, 8])
    c = Tensor.concat([a, b], 0)

    assert c.dims == [4, 2]
    assert not c.requiresGrad
    expected = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(4):
      for j in range(2):
        assert c.getitem([i, j]) == expected[i * 2 + j]

  def test_concat_dim1_rectangular(self):
    a = Tensor([2, 2], [1, 2, 3, 4])
    b = Tensor([2, 3], [10, 20, 30, 40, 50, 60])
    c = Tensor.concat([a, b], 1)

    assert c.dims == [2, 5]
    expected = [[1, 2, 10, 20, 30], [3, 4, 40, 50, 60]]
    for i in range(2):
      for j in range(5):
        assert c.getitem([i, j]) == expected[i][j]

  def test_concat_throws_on_dim_mismatch(self):
    a = Tensor([2, 2], [1, 2, 3, 4])
    b = Tensor([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9])

    with pytest.raises(ValueError):
      Tensor.concat([a, b], 0)

if __name__ == '__main__':
  raise RuntimeError("Not a standalone script")