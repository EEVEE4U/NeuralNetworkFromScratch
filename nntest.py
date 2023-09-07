import FeedForwardNeuralNetwork as nn
import functions
import numpy as np


model = nn.newRegressor(functions.MeanSquaredError)

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
model.addLayer('dense', 1, 1, 'relu')
model.train(x, y, 1000, 0.1)
print(model.predict(np.array([220])))
