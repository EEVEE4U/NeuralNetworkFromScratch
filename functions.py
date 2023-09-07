import numpy as np

"""
Matrix returned by derivative of activation function may be a square matrix 
:: 
Otherwise all Matrices should be row matrix(which will be converted to diagonal matrix)
"""

#Activations
class linear:
    def activation_function(x):
        return x
    def activation_function_derivative(x):
        return np.ones(x.shape)

class tanh:
    def activation_function(x):
        return np.tanh(x)
    def activation_function_derivative(x):
        return 1 - np.tanh(x) ** 2

class sigmoid:
    def activation_function(x):
        return 1 / (1 + np.exp(-x))
    def activation_function_derivative(x):
        return sigmoid.activation_function(x) * (1 - sigmoid.activation_function(x))

class relu:
    def activation_function(x):
        return np.maximum(0, x)
    def activation_function_derivative(x):
        return (x > 0) * 1
        
class leaky_relu:
    def activation_function(x):
        if x>=0:
            return x
        else:
            return 0.1 * x
    def activation_function_derivative(x):
        if x>=0:
            return 1
        else:
            return 0.1
        

class softmax:
    def activation_function(x):
        tmp = np.exp(x - x.max())
        return tmp / np.sum(tmp)
    def activation_function_derivative(x):
        y = softmax.activation_function(x)
        M = [y[0] for i in range(len(y[0]))]
        M = np.array(M)
        I = np.identity(M.shape[0])
        return np.multiply(M, (I - M.T))


#Errors
class MeanSquaredError:
    def get_error(y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))
    def get_gradient(y_pred, y_true):
        return 2 * (y_pred - y_true) / np.size(y_true)

class CategoricalCrossEntropy:
    def get_error(y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-11, 1-1e-11)
        log_y_pred = np.log(y_pred)
        return -1 * np.sum(np.multiply(log_y_pred, y_true))
    def get_gradient(y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-11, 1-1e-11)
        return -1 * (y_true / y_pred)

class MeanAbsoluteError:
    def get_error(y_pred, y_true):
        return np.sum(np.abs(y_pred - y_true))
    def get_gradient(y_pred, y_true):
        return np.sign(y_pred - y_true)

 #NotUsed   
class LogLoss:
    def get_error(y_pred, y_true):
        diff = np.clip(y_pred - y_true, 1e-16, 1-1e-16)
        err = np.abs(1 / np.log(np.abs(diff)))
        return np.mean(err)
    def get_gradient(y_pred, y_true):
        diff = np.clip(y_pred - y_true, 16, 1-1e-16)
        return np.sign(diff) * -1/(np.log(diff)**2) * 1/(diff) * np.sign(1/(np.log(diff)))

class TanLoss:
    def get_error(y_pred, y_true):
        diff = np.abs(y_pred - y_true)
        return np.mean(np.tan(np.pi * diff))
    def get_gradient(y_pred, y_true):
        diff = y_pred - y_true
        return 1 / np.cos(np.pi * np.abs(diff))**2 * np.pi * np.sign(diff)