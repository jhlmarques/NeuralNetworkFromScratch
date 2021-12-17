from typing import List, Callable, Union
import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(y: np.ndarray):
    return y * (1 - y)


def loss(expected: np.ndarray, output: np.ndarray):
    return np.sum(0.5 * (expected - output) ** 2)


class DenseLayer:

    cur_output: np.ndarray
    cur_delta: np.ndarray

    def __init__(self, n_of_inputs: int, n_of_neurons: int, bias: float,
                 activation_callback: Callable[[np.ndarray], np.ndarray],
                 d_activation_callback: Callable[[np.ndarray], np.ndarray]):

        self.n_of_neurons = n_of_neurons
        self.weights = np.random.uniform(size=(n_of_inputs, n_of_neurons))     # Transposed weight matrix
        self.bias = bias
        self.activation_callback = activation_callback
        self.d_activation_callback = d_activation_callback
        self.cur_output = None
        self.cur_delta = None

    def feed_forward(self, input: np.ndarray):
        self.cur_output = self.activation_callback(np.dot(input, self.weights) + self.bias)
        return self.cur_output


class NeuralNetwork:

    output: np.ndarray
    batch_input: np.ndarray
    batch_output: np.ndarray
    layers: List[DenseLayer]

    def __init__(self, input_size: int, number_of_batches: int, learn_bias: bool):
        self.input_size = input_size
        self.layer_count = 0
        self.layers = []
        self.output = None
        self.batch_input = None
        self.batch_output = None
        self.number_of_batches = number_of_batches
        self.current_batch_number = -1  # Index (see change_batch)
        self.losses = [None] * number_of_batches
        self.learn_bias = learn_bias

    # Sets the current batch
    def change_batch(self, input: Union[list, np.ndarray], output: Union[list, np.ndarray]):
        if isinstance(input, list):
            input = np.array(input)
        if isinstance(output, list):
            output = np.array(output)

        if (not isinstance(input, np.ndarray)) or (not isinstance(output, np.ndarray)):
            raise TypeError("Batch input and output must be either an array in list format or numpy array")

        self.batch_input = input
        self.batch_output = output
        self.current_batch_number += 1
        if self.current_batch_number == self.number_of_batches:
            self.current_batch_number = 0

    # Adds a layer to the network, which becomes the new output layer
    def append_layer(self, units: int, bias: float,
                     activation: Callable[[np.ndarray], np.ndarray], d_activation: Callable[[np.ndarray], np.ndarray]):

        if not len(self.layers):
            new_layer = DenseLayer(self.input_size, units, bias, activation, d_activation)
        else:
            new_layer = DenseLayer(self.layers[-1].n_of_neurons, units, bias, activation, d_activation)

        self.layers.append(new_layer)
        self.layer_count += 1

    # Feeds forward the inputs
    def feed_forward(self, log=False):
        prev_output = self.batch_input
        cur_output = float()
        for layer in self.layers:
            cur_output = layer.feed_forward(prev_output)
            if log:
                print(f"-----<feed_forward>------\nInput:\n{prev_output}\n"
                      f"Output:\n{cur_output}\n-----</feed_forward>-----")
            prev_output = cur_output
        self.output = cur_output

    # Calculates the deltas
    # noinspection PyUnboundLocalVariable
    def calculate_deltas(self, log=False):

        for j in reversed(range(len(self.layers))):
            layer = self.layers[j]
            last_delta: np.ndarray
            if j == self.layer_count - 1:   # Output layer
                error = (self.batch_output - self.output)
            else:   # Hidden layers
                error = (last_delta * self.layers[j+1].weights.T)

            last_delta = error * layer.d_activation_callback(layer.cur_output)
            layer.cur_delta = last_delta

            if log:
                print(f"New delta:\n{last_delta}")

    # #Uses the delta values of each layer to adjust weights and bias
    def adjust_weights(self, l_rate: float, log=False):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                grad_weights = self.batch_input.T @ layer.cur_delta
            else:
                grad_weights = self.layers[i-1].cur_output.T @ layer.cur_delta

            if self.learn_bias:
                grad_bias = np.sum(layer.cur_delta)

            if log:
                print(f"Adjusting weight with:\n{grad_weights}\n")
                if self.learn_bias:
                    print(f"Adjusting bias with:\n{grad_bias}:")

            layer.weights += grad_weights * l_rate
            layer.bias += grad_bias * l_rate

            if log:
                print(f"Result:\nNew weights:\n{layer.weights}")
                if self.learn_bias:
                    print(f"New bias: {layer.bias}\n")

    # Uses the previously defined functions to train the network with a batch
    def train_batch(self, learning_rate, log_functions, log_comparison):
        self.feed_forward(log_functions)
        self.calculate_deltas(log_functions)
        self.adjust_weights(learning_rate, log_functions)
        self.losses[self.current_batch_number] = (loss(self.batch_output, self.output))
        if log_comparison:
            print(f"Batch {self.current_batch_number+1:02}:\n"
                  f"Expected\t---->\tOutput\n"
                  f"==========================")
            for y, y_hat in zip(self.batch_output, self.output):
                print(f"{y}\t\t{y_hat}")
