import RecurrentNeuralNetwork as rnn
import numpy as np
model = rnn.Many2Many(3, rnn.functions.tanh, rnn.functions.tanh, rnn.functions.MeanAbsoluteError)
x = np.array([
    [[[1,2,4]], [[1,2,74]]],
    [[[1,2,94]], [[1,2,44]]],
    [[[11,2,4]], [[1,42,4]]]
]
)
y = np.array([[[0, 1, 1]], [[1, 0, 1]]])

print(model.train(x, y, 2))