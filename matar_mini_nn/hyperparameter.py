from itertools import product
from .core.net_init import NeuralNetwork
from .trainer import Trainer

class Hyperparamt:
    def __init__(self, optimizer_class, search_space, input_size, output_size):
        self.optimizer_class = optimizer_class
        self.search_space = search_space
        self.input_size = input_size
        self.output_size = output_size
        self.best_params = None
        self.best_accuracy = -float("inf")

    def _generate_configs(self):
        keys = self.search_space.keys()
        values = self.search_space.values()
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def find(self, x_train, t_train, x_val, t_val):
        for params in self._generate_configs():
            network = NeuralNetwork(
                input_size=self.input_size,
                hidden_sizes=params["hidden_sizes"],
                output_size=self.output_size,
                activation=params["activation"],
                dropout_ratio=params["dropout"]
            )

            optimizer = self.optimizer_class(lr=params["learning_rate"])

            trainer = Trainer(network, optimizer)
            trainer.fit(
                x_train, t_train,
                x_test=x_val, t_test=t_val,
                epochs=params["epochs"],
                batch_size=params["batch_size"]
            )

            acc = network.accuracy(x_val, t_val)

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_params = params

        return self.best_params, self.best_accuracy
