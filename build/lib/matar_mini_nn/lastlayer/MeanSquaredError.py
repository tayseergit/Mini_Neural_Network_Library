import numpy as np


class MeanSquaredError:
    def forward(self, x, t):
 
        self.x = x
        self.t = t
        batch_size = x.shape[0]
        loss = 0.5 * np.sum((x - t) ** 2) / batch_size
        return loss

    def backward(self, dout=1):
        batch_size = self.x.shape[0]
        dx = (self.x - self.t) * dout / batch_size
        return dx
