from torch import nn, Tensor

class ClassifierHead(nn.Module):
    def __init__(self, d_model: int = 64, d_hidden: int = 512, num_classes: int = 107):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_hidden)
        self.nonlinear = nn.GELU()
        self.f2 = nn.Linear(d_hidden, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.nonlinear(self.f1(x))
        return self.f2(e1)