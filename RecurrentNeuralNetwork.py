import layers
import functions
import numpy as np
import pickle

class RNN:
    def __init__(self, rnn_outputs: "all / one", n_inputs):
        self.n_inputs = n_inputs
        self.rnn_outputs = rnn_outputs
        self.state = 0

    def forward(self, previous_output: np.ndarray):
        f