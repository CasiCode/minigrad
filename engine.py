import math
import numbers


# TODO: Implement graph viewer
# TODO: Implement complex numbers
class Scalar:
    def __init__(self, value, _children=(), _operation='') -> None:
        assert isinstance(value, numbers.Real), 'only numeric datatypes are supported'
        self.value = value
        self.grad = 0.0
        self._prev = set(_children)
        self._operation = _operation
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Scalar(value={self.value})'

    def __add__(self, rhs):
        rhs = rhs if isinstance(rhs, Scalar) else Scalar(rhs)
        out = Scalar(
            value=self.value+rhs.value,
            _children=(self, rhs),
            _operation='__add__'
        )

        def _backward():
            self.grad += out.grad
            rhs.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, rhs):
        return self + rhs

    def __mul__(self, rhs):
        rhs = rhs if isinstance(rhs, Scalar) else Scalar(rhs)
        out = Scalar(
            value=self.value*rhs.value,
            _children=(self, rhs),
            _operation='__mul__'
        )

        def _backward():
            self.grad += rhs.value * out.grad
            rhs.grad += self.value * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, rhs):
        return self * rhs

    def __neg__(self):
        return self * -1

    def __sub__(self, rhs):
        return self + (-rhs)

    def __rsub__(self, rhs):
        return rhs + (-self)

    def __pow__(self, power):
        power = power if isinstance(power, Scalar) else Scalar(power)
        assert not(self.value <= 0.0 and (power.value != int(power.value))), \
            'Non-postitive value to the power of non-int. Not a smooth function'
        assert not(self.valur == 0.0 and power.value < 0.0), \
            'Zero to the power of negative. Not a smooth function'

        out = Scalar(
            value=self.value**power.value,
            _children=(self, power),
            _operation='__pow__'
        )

        def _backward():
            self.grad += out.value * (power.value/self.value) * out.grad
            power.grad += out.value * math.log(self.value) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, rhs):
        assert rhs.value != 0.0, 'cannot divide by zero'
        return self * rhs**-1

    def __rtruediv__(self, rhs):
        return rhs * self**-1

    def relu(self):
        out = Scalar(
            value=self.value if self.value > 0.0 else 0.0,
            _children=(self, ),
            _operation='relu'
        )

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v in visited: pass
            visited.add(v)
            for child in v.prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
