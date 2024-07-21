import torch
from torch.autograd import Variable
from torch import nn
from d2l import torch as d2l
# uzing lazylinear
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    def forward(self, X):
        return self.net(X)
    def loss(self, y, y_hat):
        fn = nn.MSELoss()
        return fn(y, y_hat)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)