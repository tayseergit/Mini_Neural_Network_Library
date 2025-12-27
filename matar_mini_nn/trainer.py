import numpy as np


class Trainer:
 
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer

        self.train_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def train_steps(self, x_batch, t_batch):
 
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_history.append(loss)
        return loss

    def fit(
        self,
        x_train,
        t_train,
        x_test=None,
        t_test=None,
        epochs=1,
        batch_size=32,
    ):
 
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(train_size)
            x_train = x_train[shuffle_idx]
            t_train = t_train[shuffle_idx]

            for i in range(iter_per_epoch):
                batch_mask = slice(i*batch_size, (i+1)*batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]
                self.train_steps(x_batch, t_batch)

            train_acc = self.network.accuracy(x_train, t_train)
            self.train_acc_history.append(train_acc)

            if x_test is not None and t_test is not None:
                test_acc = self.network.accuracy(x_test, t_test)
                self.test_acc_history.append(test_acc)
            else:
                test_acc = None
