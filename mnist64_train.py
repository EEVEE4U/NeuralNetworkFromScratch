
import FeedForwardNeuralNetwork as nn

import numpy as np
from sklearn.datasets import load_digits
mnist = load_digits()

x_train, y_train = mnist["data"], mnist["target"]
x_train = x_train.reshape(len(x_train) ,1, len(x_train[0]))
model = nn.newClassifier(nn.functions.CategoricalCrossEntropy)
model.addLayer(layer_type='shorten', n_inputs=64, n_neurons=20)
model.addLayer(layer_type='dense', n_inputs=20, n_neurons=10, activation='relu')
model.addLayer(layer_type='dense', n_inputs=10, n_neurons=10, activation='relu')
model.addLayer(layer_type="dense", n_inputs=10, n_neurons=10, activation='softmax')

model.train(x_train, y_train, 100, 0.001)