import numpy as np

class SoftmaxCrossEntropy:

    def forward(self, x, t):
     
        self.t = t
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        batch_size = x.shape[0]
        self.loss = -np.sum(t * np.log(self.y + 1e-7)) / batch_size
        return self.loss

    def backward(self, dout=1):
  
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
