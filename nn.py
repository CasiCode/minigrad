import random
from minigrad.engine import Scalar


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, input_num, nonlin=True):
        self.w = [
            Scalar(value=random.uniform(-1, 1)) for _ in range(input_num)
        ]
        self.b = Scalar
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron ({len(self.w)})"


class Layer(Module):
    def __init__(self, input_num, output_num, **kwargs):
        self.neurons = [Neuron(input_num, **kwargs) for _ in output_num]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"


class MLP(Module):
    def __init__(self, input_num, output_nums):
        sizes = [input_num] + output_nums
        self.layers = [
            Layer(sizes[i], sizes[i+1], nonlin=i!=len(output_nums)-1)
            for i in range(len(output_nums))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
