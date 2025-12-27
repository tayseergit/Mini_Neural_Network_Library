# i make the libray to build the layer from hyperparameter outputs
# not from choose each layer dependently
# so you need to init the parameter then ge the best of them tehm tran the net
# i know that not is the task 


# ## Usage
#  first handle the parameterr you need to use 
#  for example :

# parameter_init = {
#     "hidden_sizes": [[64, 32], [128, 64]],
#     "activation": ["relu"],
#     "dropout": [0.2],
#     "learning_rate": [0.1],
#     "batch_size": [32, 64],
#     "epochs": [1, 2]
# }

# # Hyperparameter call for get the best parameter for taraining 

# hyper = Hyperparamt(
#     optimizer_class=SGD,
#     search_space=parameter_init,
#     input_size=x_train.shape[1],
#     output_size=t_train.shape[1]
# )
# best_params, best_acc = hyper.find(x_train, t_train, x_test, t_test)


# # build network with best hyperparameters
# network = NeuralNetwork(
#     input_size=x_train.shape[1],
#     hidden_sizes=best_params["hidden_sizes"],
#     output_size=t_train.shape[1],
#     activation=best_params["activation"],
#     dropout_ratio=best_params["dropout"]
# )

# # optimizer and trainer
# optimizer = SGD(lr=best_params["learning_rate"])
# trainer = Trainer(network, optimizer)

# # train the network
# trainer.fit(
#     x_train, t_train,
#     x_test=x_test, t_test=t_test,
#     epochs=best_params["epochs"],
#     batch_size=best_params["batch_size"]
# )
 