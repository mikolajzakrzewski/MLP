import numpy as np


class Neuron:
    def __init__(self, input_values, weights=None, bias=0):
        self.input_values = input_values
        if weights is None:
            self.weights = np.random.randn(input_values.shape[0], 1)
        else:
            self.weights = weights

        self.bias = bias
        self.net_sum = self.calculate_net_sum()
        self.output = self.calculate_output()

    def calculate_net_sum(self):
        return np.sum(np.multiply(self.input_values, self.weights)) + self.bias

    def calculate_output(self):
        return np.tanh(self.net_sum)

    def update(self, input_values=None, weights=None, bias=None):
        if input_values is not None:
            self.input_values = input_values

        if weights is not None:
            self.weights = weights

        if bias is not None:
            self.bias = bias

        if self.input_values is not None or self.weights is not None or self.bias is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layers_num, hidden_layers_sizes, output_layer_size):
        self.input_layer_size = input_layer_size
        self.input_layer = np.zeros(self.input_layer_size, dtype=Neuron)
        self.hidden_layers_num = hidden_layers_num
        self.hidden_layers_sizes = hidden_layers_sizes
        self.hidden_layers = [np.zeros(hidden_layers_sizes[i], dtype=Neuron) for i in range(len(hidden_layers_sizes))]
        self.output_layer_size = output_layer_size
        self.output_layer = np.zeros(self.output_layer_size, dtype=Neuron)

    def feedforward(self, input_values):
        input_layer_outputs = np.zeros(self.input_layer_size)
        for i in range(self.input_layer_size):
            self.input_layer[i] = Neuron(input_values)
            input_layer_outputs[i] = self.input_layer[i].output

        hidden_layer_inputs = input_layer_outputs
        for i in range(self.hidden_layers_num):
            hidden_layer_outputs = np.zeros(self.hidden_layers_sizes[i])
            for j in range(self.hidden_layers_sizes[i]):
                self.hidden_layers[i][j] = Neuron(hidden_layer_inputs)
                hidden_layer_outputs[i] = self.hidden_layers[i][j].output

            hidden_layer_inputs = hidden_layer_outputs

        output_layer_inputs = hidden_layer_outputs
        output_layer_outputs = np.zeros(self.output_layer_size)
        for i in range(self.output_layer_size):
            self.output_layer[i] = Neuron(output_layer_inputs)
            output_layer_outputs[i] = self.output_layer[i].output

        return output_layer_outputs

    def backpropagation(self, output_values, learning_rate):
        # TODO: implement backpropagation
        return None

    def train(self, input_values, learning_rate, epochs_num=None, accuracy=None):
        # TODO: implement NN training
        return None


if __name__ == '__main__':
    neural_network = NeuralNetwork(2, 2, [2, 2], 2)
    print(neural_network.feedforward(np.array([2, 2])))
