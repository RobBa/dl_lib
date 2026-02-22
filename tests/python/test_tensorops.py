"""
Robert Baumgartner, r.baumgartner-1@tudelft.nl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python_lib"))

from dl_lib import Tensor, Device

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

if __name__ == '__main__':
    raise RuntimeError("Not a standalone script")