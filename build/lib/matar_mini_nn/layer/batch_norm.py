import numpy as np


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, eps=1e-7):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.eps = eps

        self.running_mean = np.zeros_like(gamma)
        self.running_var = np.zeros_like(gamma)

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        if train_flg:
            mu = x.mean(axis=0)
            var = x.var(axis=0)

            self.xc = x - mu
            self.std = np.sqrt(var + self.eps)
            xn = self.xc / self.std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xn = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xc / self.std * dout, axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
