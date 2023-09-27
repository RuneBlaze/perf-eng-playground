from torch.utils.data import IterableDataset
import torch
import math
from torch import Tensor, nn
from typing import Iterator, TypedDict
from functools import cache
from .utils import thread_map
from viztracer import get_tracer

class FibDataPoint(TypedDict):
    x: Tensor
    y: Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 64, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.reshape(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        tracer = get_tracer()
        if tracer:
            tracer.enable = True
            tracer.log_var("sin_x", math.sin(x))
            tracer.enable = False
        return self.pe[x]
    
@cache
def fib_memoized(n: int) -> int:
    if n < 2:
        return n
    return fib_memoized(n - 1) + fib_memoized(n - 2)

def fib_naive(n: int) -> int:
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

class FibDataset(IterableDataset):
    """Classification dataset, given x, classify f(x) % 107.
    x is encoded as using positional encoding."""
    def __init__(self):
        self.pe = PositionalEncoding()

    def __iter__(self) -> Iterator[FibDataPoint]:
        while True:
            n = 5000
            perm = torch.randperm(n)
            for i, x in thread_map(lambda i: (i, self.pe(perm[i].item())), range(n)):
                yield {'x': x, 'y': fib_memoized(i) % 107}