import torch
import d2l
class SGD(d2l.HyperParameters):
    """
    Minibatch SGD
    """
    def __init__(self, params, lr):
        self.save_hyperparameters()
    
    def step(self):
        for param in self.params:
            # move param closer w/ SGD
            param -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                # reset grad to perform backprop
                param.grad.zero_()