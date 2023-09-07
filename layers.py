import numpy as np

# fully connected dense layer
class DenseLayer:
    def __init__ (self, n_inputs: int, n_neurons: int) -> None:
        #set up weights and biases of layer
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.weights = np.random.rand(self.n_inputs, self.n_neurons) - 0.5
        self.bias = np.random.rand(1, self.n_neurons) - 0.5

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        #adjust weights, bias and return input_gradient        
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        bias_gradient = output_gradient
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights = self.weights - learning_rate * weights_gradient
        self.bias = self.bias - learning_rate * bias_gradient
        return input_gradient


# seperate layer for applying activation function
class ActivationLayer:
    #ActivationLayer behaves as seperate layer which only applies activation_function
    def __init__(self, ActivationClass):
        self.ActivationClass = ActivationClass
        self.activation_function = self.ActivationClass.activation_function
        self.activation_function_derivative = self.ActivationClass.activation_function_derivative

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return self.activation_function(self.inputs)

    def backward(self, output_gradient: np.ndarray, learning_rate: "Not required") -> np.ndarray:
        #return input_gradient
        inputs_derivative = self.activation_function_derivative(self.inputs)
        if inputs_derivative.shape[0] != inputs_derivative.shape[1] :#check if not square matrix to convert to diagonal
            inputs_derivative = np.diag(inputs_derivative[0])

        input_gradient = np.dot(output_gradient, inputs_derivative)

        return input_gradient


# reduce the number of inputs for special case by using not fully connected layer
class ShortLayer:
    def __init__ (self, n_inputs: int, n_neurons: int) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.inputs_for_each_neuron = int(np.ceil(n_inputs/ n_neurons))
        

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        outputs = []
        for i in range(self.n_neurons):
            outputs.append(np.sum(self.inputs[0][i*self.inputs_for_each_neuron : (i+1)*self.inputs_for_each_neuron]))

        outputs = np.array(outputs).reshape(1, len(outputs))
        return outputs

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        input_gradient = []
        rm_from_last = self.n_inputs % self.inputs_for_each_neuron
        for i in output_gradient[0]:
            for j in range(self.inputs_for_each_neuron):
                input_gradient.append(i)
        for i in range(rm_from_last):
            input_gradient.pop()
        input_gradient = np.array(input_gradient)
        return input_gradient
    

    
    
    

    

