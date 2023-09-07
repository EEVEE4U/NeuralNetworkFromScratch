import layers
import functions
import pickle
import numpy as np


"""Use Many Layers but use less Neurons in each layer"""
#In case accuracy does not increases, lower the learning rate

class newClassifier:
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.layers = []

    def addLayer(self, layer_type: str, n_inputs: int, n_neurons: int, activation: str) -> None:
        if layer_type == "dense":
            self.layers.append(layers.DenseLayer(n_inputs, n_neurons))
        elif layer_type == "shorten":
            self.layers.append(layers.ShortLayer(n_inputs, n_neurons))
        else:
            raise Exception(f"No layer named {layer_type}")


        if activation == 'relu':
            self.layers.append(layers.ActivationLayer(functions.relu))
        elif activation == 'softmax':
            self.layers.append(layers.ActivationLayer(functions.softmax))
        elif activation == 'tanh':
            self.layers.append(layers.ActivationLayer(functions.tanh))
        elif activation == 'sigmoid':
            self.layers.append(layers.ActivationLayer(functions.sigmoid))
        elif activation == 'linear':
            self.layers.append(layers.ActivationLayer(functions.linear))
        
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X_train: "array of row matrices", Y_train: "one hot encoded", epochs: int, learning_rate: float, showError=True, showAccuracy=True, minimum_accuracy_percent=101) -> "error_gradient of last layer":
        if len(Y_train.shape) == 1:
            Y_train = np.array([[0 if i!=int(j) else 1 for i in range(self.layers[-2].n_neurons)] for j in Y_train])
        
        #Training
        for epoch in range(epochs):
            mean_error = 0
            correct = 0
            for x, y in zip(X_train, Y_train):
                prediction = self.predict(x)
                mean_error += self.loss_function.get_error(prediction, y)
                error_gradient = self.loss_function.get_gradient(prediction, y)
                for layer in reversed(self.layers):
                    error_gradient = layer.backward(error_gradient, learning_rate)
                if np.argmax(prediction) == np.argmax(y):
                    correct += 1

            accuracy = correct / len(Y_train) * 100
        
            print(f"epoch: {epoch+1}")
            if showError==True:
                print(f"Mean Error: {mean_error}")
            if showAccuracy==True:
                print(f"Accuracy: {accuracy}%")
            if accuracy >= minimum_accuracy_percent:
                break
        return error_gradient
            




class newRegressor:
    #Remember to have 1 neuron in last layer output
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.layers = []

    def addLayer(self, layer_type: str, n_inputs: int, n_neurons: int, activation: str = None) -> None:
        if layer_type == "dense":
            self.layers.append(layers.DenseLayer(n_inputs, n_neurons))
        elif layer_type == "shorten":
            self.layers.append(layers.ShortLayer(n_inputs, n_neurons))
        else:
            raise Exception(f"No layer named {layer_type}")


        if activation == 'relu':
            self.layers.append(layers.ActivationLayer(functions.relu))
        elif activation == 'softmax':
            self.layers.append(layers.ActivationLayer(functions.softmax))
        elif activation == 'tanh':
            self.layers.append(layers.ActivationLayer(functions.tanh))
        elif activation == 'sigmoid':
            self.layers.append(layers.ActivationLayer(functions.sigmoid))
        elif activation == 'linear':
            self.layers.append(layers.ActivationLayer(functions.linear))


    def predict(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X_train: "array of row matrices", Y_train: "one hot encoded", epochs: int, learning_rate: float, showError=True, showAccuracy=True) -> None:
        #Training
        for epoch in range(epochs):
            accuracy = 0
            mean_error = 0
            for x, y in zip(X_train, Y_train):
                prediction = self.predict(x)
                mean_error += self.loss_function.get_error(prediction, y) / len(Y_train)
                error_gradient = self.loss_function.get_gradient(prediction, y)
                accuracy += (prediction - y)/np.size(Y_train) * 100
                for layer in reversed(self.layers):
                    error_gradient = layer.backward(error_gradient, learning_rate)
        

            print(f"epoch: {epoch+1}")
            if showError==True:
                print(f"Mean Error: {mean_error}")
            if showAccuracy==True:
                print(f"Approximated to:  {accuracy} accuracy")
            if epoch % 5 == 0: #decrease learning rate timely to avoid looping through same accuracy
                learning_rate = learning_rate/1.1




#For loading and saving model
def save_model(filename: str, model):
    with open(f"{filename}.pkl", "wb") as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def load_model(filename: str):
    with open(f"{filename}.pkl", "rb") as inp:
        model = pickle.load(inp)
    return model
