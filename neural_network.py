import numpy as np


class Neuron:
    def __init__(self, input_size, input_values=None, weights=None, bias=0):
        self.input_values = input_values
        if weights is None:
            self.weights = np.random.randn(input_size, 1)
        else:
            self.weights = weights

        self.bias = bias
        if input_values is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()
        else:
            self.net_sum = 0
            self.output = 0

    def calculate_net_sum(self):
        return np.sum(np.multiply(self.input_values, self.weights)) + self.bias

    def calculate_output(self):
        return np.tanh(self.net_sum)

    def calculate_output_derivative(self):
        return 1 - np.tanh(self.net_sum) ** 2

    def update(self, input_values=None, weights=None, bias=None):
        if input_values is not None:
            self.input_values = input_values

        if weights is not None:
            self.weights = weights

        if bias is not None:
            self.bias = bias

        if self.input_values is not None and self.weights is not None and self.bias is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layers_num, hidden_layers_sizes, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layers_num = hidden_layers_num
        self.hidden_layers_sizes = hidden_layers_sizes
        self.hidden_layers = [np.empty(hidden_layer_size, dtype=Neuron) for hidden_layer_size in hidden_layers_sizes]
        neuron_input_size = input_layer_size
        for hidden_layer in self.hidden_layers:
            hidden_layer.fill(Neuron(neuron_input_size))
            neuron_input_size = len(hidden_layer)

        self.output_layer_size = output_layer_size
        self.output_layer = np.empty(self.output_layer_size, dtype=Neuron)
        self.output_layer.fill(Neuron(neuron_input_size))

    def feedforward(self, input_values):
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.update(input_values, weights=None, bias=None)

            input_values = [neuron.output for neuron in hidden_layer]

        for neuron in self.output_layer:
            neuron.update(input_values, weights=None, bias=None)

        output_values = [neuron.output for neuron in self.output_layer]
        return output_values

    def backpropagation(self, output_values):
        gradients = []
        output_layer_gradient = [
            (neuron.output - output_values[i]) *
            neuron.calculate_output_derivative() *
            neuron.input_values
            for i, neuron in enumerate(self.output_layer)
        ]
        output_layer_gradient = np.array(output_layer_gradient)
        gradients.append(output_layer_gradient)
        gradient_above = output_layer_gradient
        layer_above = self.output_layer
        for hidden_layer in reversed(self.hidden_layers):
            hidden_layer_gradient = []
            # TODO: Fix derivative calculation
            layer_above_derivatives = np.array([neuron.calculate_output_derivative() for neuron in layer_above])
            layer_above_weights = np.array([neuron.weights[:, 0] for neuron in layer_above])
            for neuron in hidden_layer:
                derivative = np.sum(gradient_above * layer_above_derivatives * layer_above_weights)
                error = derivative * neuron.calculate_output_derivative() * neuron.input_values
                hidden_layer_gradient.append(error)

            gradients.append(hidden_layer_gradient)
            gradient_above = hidden_layer_gradient
            layer_above = hidden_layer

        return gradients[::-1]
