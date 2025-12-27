import numpy as np

from core.net_init import NeuralNetwork
from hyperparameter import Hyperparamt
from trainer import Trainer
from optimizers.sgd import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


X, y = fetch_openml('mnist_784', return_X_y=True,as_frame = False)
x_train,x_test,t_train,t_test = train_test_split(X, y, test_size=10000, shuffle=False)
t_train = t_train.astype(int)
t_test = t_test.astype(int)
encoder = OneHotEncoder(sparse_output=False)
t_train = encoder.fit_transform(t_train.reshape(-1, 1))
t_test = encoder.transform(t_test.reshape(-1, 1))
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255.0
x_test /= 255.0


search_space = {
    "hidden_sizes": [[64, 32], [128, 64]],
    "activation": ["relu",],
    "dropout": [0.2,],
    "learning_rate": [ 0.1],
"batch_size": [32, 64],
    "epochs": [1,2]  
}


hyper =Hyperparamt(
    optimizer_class=SGD,
    search_space=search_space,
    input_size=x_train.shape[1],
    output_size=t_train.shape[1]
)
best_params, best_acc = hyper.find(x_train, t_train, x_test, t_test)

network = NeuralNetwork(
  input_size=x_train.shape[1],
    hidden_sizes=best_params["hidden_sizes"],
    output_size=t_train.shape[1],
    activation=best_params["activation"],
    dropout_ratio=best_params["dropout"]
)

optimizer = SGD(lr=best_params["learning_rate"])
trainer = Trainer(network, optimizer)

trainer.fit(
    x_train, t_train,
    x_test=x_test, t_test=t_test,
    epochs=best_params["epochs"],              
    batch_size=best_params["batch_size"]
)

train_acc = network.accuracy(x_train, t_train)
test_acc = network.accuracy(x_test, t_test)

print("\n=== Test Results ===")
print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)