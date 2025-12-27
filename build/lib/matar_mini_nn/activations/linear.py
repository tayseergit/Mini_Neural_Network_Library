class Linear:
    def forward(self, x):
        self.x = x
        return x

    def backward(self, dout):
        dx = dout
        return dx
