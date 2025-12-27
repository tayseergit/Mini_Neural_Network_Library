import numpy as np
from collections import OrderedDict
from ..activations.linear import *
from ..layer.affine import *
from ..lastlayer.softmaxCE import *
from ..layer.dropout import *
from ..activations.relu import ReLU
from ..activations.sigmoind import Sigmoid

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout_ratio=0.0):
        self.params = {}
        self.layers = OrderedDict()
        self.dropout_ratio = dropout_ratio
        
        self._init_weights(input_size, hidden_sizes, output_size, activation)
        
        self._build_network(hidden_sizes, activation)
        
        self.lastLayer = SoftmaxCrossEntropy()

    def _init_weights(self, input_size, hidden_sizes, output_size, activation):
        all_sizes = [input_size] + hidden_sizes + [output_size]
        
        for idx in range(1, len(all_sizes)):
            scale = np.sqrt(2.0 / all_sizes[idx-1]) if activation == 'relu' else 0.01
            
            self.params['W' + str(idx)] = scale * np.random.randn(all_sizes[idx-1], all_sizes[idx])
            self.params['b' + str(idx)] = np.zeros(all_sizes[idx])

    def _build_network(self, hidden_sizes, activation):
        num_hidden = len(hidden_sizes)
        
        for idx in range(1, num_hidden + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])            
            self.layers['Activation' + str(idx)] = self._get_activation_layer(activation)
            
            if self.dropout_ratio > 0:
                self.layers['Dropout' + str(idx)] = Dropout(self.dropout_ratio)
        
        final_idx = num_hidden + 1
        self.layers['Affine' + str(final_idx)] = Affine(self.params['W' + str(final_idx)], self.params['b' + str(final_idx)])

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1) 
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, len(self.params)//2 + 1):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
            
        return grads

    def _get_activation_layer(self, name):
        if name.lower() == 'relu': return ReLU()
        if name.lower() == 'sigmoid': return Sigmoid()
        if name.lower() == 'linear': return Linear()
        return None