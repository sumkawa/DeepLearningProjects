import torch
from d2l import torch as d2l
from optimization import SGD
class LinearRegression(d2l.Module):
    """
    Linear regression model from scratch.
    """
    def __init__(self, num_inputs, lr, sigma=0.01):
        """
        Initializes LinearRegression model.

        Params
        ______
        num_inputs: int
            Number input features
        lr: float
            Learning rate for model (eta)
        sigma: float, optimal
            Default 0.01. Std of normal distribution used to initialize weights.
        """
        super().__init__()
        self.save_hyperparameters()
        # initialize d X 1 column vector w w/ random weights
        # using mu = 0, sigma = 0.01 
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    def forward(self, X):
        # simple forward pass in linear regression, y = wx + b
        return torch.matmul(X, self.w) + self.b
    def loss(self, y_hat, y):
        l = (y_hat-y)**2 /2 # divide by two for easier differentiation
        return l.mean() # average loss over all training examples
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)