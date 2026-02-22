"""
Robert Baumgartner, r.baumgartner-1@tudelft.nl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python_lib"))
print(sys.path)

from dl_lib import Tensor
import pytest

class TestAutograd:
    def test_backward(self):
        t = Tensor([2, 2], True)

        loss = (t * 2).sum()
        loss.backward()
        
        assert t.grads is not None

    def test_nograd_throws(self):
        t1 = Tensor([1], [3.0], False)
        t2 = Tensor([1], [3.0], False)

        t3 = t1*t2

        assert not t3.requiresGrad
        with pytest.raises(RuntimeError):
            t3.backward()

    def test_add(self):
        t1 = Tensor([1], [3.0], True)
        t2 = Tensor([1], [2.0], True)
        
        t3 = t1+t2
        loss = t3*t3

        loss.backward()

        assert t1.grads.getitem(0) == pytest.approx(10.0)
        assert t2.grads.getitem(0) == pytest.approx(10.0)

    def test_scalar_mul(self):
        t1 = Tensor([1], [2.0], True)
        t2 = Tensor([1], [3.0], True)

        t3 = t1*t2
        loss = t3*t3

        loss.backward()

        assert t1.grads.getitem(0) == pytest.approx(36.0)
        assert t2.grads.getitem(0) == pytest.approx(24.0)

    def test_matmul(self):
        t1 = Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
        t2 = Tensor([3, 2], [1, 2, 3, 4, 5, 6], True)

        t3 = t1@t2
        loss = t3.sum()

        loss.backward()

        # dL/dt1 = dloss/dt3 @ t2^t = Ones({2, 2}) @ t2^t
        assert t1.grads.getitem([0, 0]) == pytest.approx(3.0)
        assert t1.grads.getitem([0, 1]) == pytest.approx(7.0)
        assert t1.grads.getitem([0, 2]) == pytest.approx(11.0)
        assert t1.grads.getitem([1, 0]) == pytest.approx(3.0)
        assert t1.grads.getitem([1, 1]) == pytest.approx(7.0)
        assert t1.grads.getitem([1, 2]) == pytest.approx(11.0)

        # dL/dt2 = t1^t @ dloss/dt3 = t1^t @ Ones({2, 2})
        assert t2.grads.getitem([0, 0]) == pytest.approx(5.0)
        assert t2.grads.getitem([0, 1]) == pytest.approx(5.0)
        assert t2.grads.getitem([1, 0]) == pytest.approx(7.0)
        assert t2.grads.getitem([1, 1]) == pytest.approx(7.0)
        assert t2.grads.getitem([2, 0]) == pytest.approx(9.0)
        assert t2.grads.getitem([2, 1]) == pytest.approx(9.0)

    def test_chainrule(self):
        x = Tensor([1], [2.0], True)

        y = x * x
        z = x + y
        loss = z * z

        loss.backward()

        # dloss/dx = 2(x^2 + x) * (2x + 1)
        # At x=2: 2(4 + 2) * (4 + 1) = 2 * 6 * 5 = 60
        assert x.grads.getitem(0) == pytest.approx(60.0)

    def test_multivariate_chainrule(self):
        x = Tensor([2], [1.0, 2.0], True)
        y = x * 3
        
        loss = Tensor([1], [0.0], True)
        for i in range(len(y)):
            loss = loss + y[i]
        loss.backward()

        assert x.grads.getitem(0) == pytest.approx(3.0)
        assert x.grads.getitem(1) == pytest.approx(3.0)

        assert y.grads.getitem(0) == pytest.approx(1.0)
        assert y.grads.getitem(1) == pytest.approx(1.0)

if __name__ == '__main__':
    raise RuntimeError("Not a standalone script")