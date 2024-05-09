import numpy as np
import random
import file_utils as fu


class Neuron:
    def __init__(self, input_size, input_values=None, weights=None, include_bias=True):
        self.input_values = input_values
        if weights is None:
            self.weights = np.array(0.5 * np.random.randn(input_size))
        else:
            self.weights = weights

        self.include_bias = include_bias
        if include_bias:
            self.bias = random.uniform(-0.5, 0.5)
        else:
            self.bias = 0.0

        if input_values is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()
        else:
            self.net_sum = 0
            self.output = 0

    def calculate_net_sum(self):
        return np.dot(self.input_values, self.weights) + self.bias

    def calculate_output(self):
        return 1 / (1 + np.exp(-self.net_sum))

    def calculate_output_derivative(self):
        return self.output * (1 - self.output)

    def update(self, input_values=None, weights=None, bias=None):
        if input_values is not None:
            self.input_values = np.array(input_values)

        if weights is not None:
            self.weights = np.array(weights)

        if bias is not None and self.include_bias:
            self.bias = bias

        if self.input_values is not None and self.weights is not None and self.bias is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layers_num, hidden_layers_sizes, output_layer_size, include_bias=True):
        self.input_layer_size = input_layer_size
        self.hidden_layers_num = hidden_layers_num
        neuron_input_size = input_layer_size
        layers_sizes = hidden_layers_sizes
        layers_sizes.append(output_layer_size)
        self.layers_sizes = layers_sizes
        self.layers = [np.empty(layer_size, dtype=Neuron) for layer_size in layers_sizes]
        for layer in self.layers:
            layer_size = len(layer)
            for i in range(layer_size):
                layer[i] = Neuron(neuron_input_size, include_bias=include_bias)

            neuron_input_size = layer_size

        self.include_bias = include_bias

    def feedforward(self, values):
        values = np.array(values)
        for layer in self.layers:
            for neuron in layer:
                neuron.update(values)

            values = [neuron.output for neuron in layer]

        return np.array(values)

    def calculate_total_error(self, output_values):
        total_err = 0
        for i, neuron in enumerate(self.layers[-1]):
            total_err += ((neuron.output - output_values[i]) ** 2) / 2

        total_err /= self.layers_sizes[-1]
        return total_err

    def backpropagation(self, output):
        weight_grads = []
        bias_grads = []
        error_signals = np.array([
            (neuron.output - output[i]) * neuron.calculate_output_derivative()
            for i, neuron in enumerate(self.layers[-1])
        ])
        errors = np.array([
            error_signals[i] * neuron.input_values
            for i, neuron in enumerate(self.layers[-1])
        ])
        weight_grads.append(errors)
        bias_grads.append(error_signals)
        grad_above = errors
        layer_above = self.layers[-1]
        for hidden_layer in reversed(self.layers[:-1]):
            layer_above_weights = np.array([neuron.weights for neuron in layer_above])
            error_signals = np.array([
                np.dot(layer_above_weights[:, i], grad_above[:, i]) * neuron.calculate_output_derivative()
                for i, neuron in enumerate(hidden_layer)
            ])
            errors = np.array([
                error_signals[i] * neuron.input_values
                for i, neuron in enumerate(hidden_layer)
            ])
            weight_grads.append(errors)
            bias_grads.append(error_signals)
            grad_above = errors
            layer_above = hidden_layer

        return weight_grads[::-1], bias_grads[::-1]

    def adjust(self, weight_grads, prev_weight_grads, bias_grads, prev_bias_grads, learning_rate, momentum):
        for i, layer in enumerate(self.layers):
            weight_errs = np.array(weight_grads[i])
            weights = np.array([neuron.weights for neuron in layer])
            weights -= learning_rate * weight_errs
            bias_errs = np.array(bias_grads[i])
            biases = np.array([neuron.bias for neuron in layer])
            biases -= learning_rate * bias_errs
            if prev_weight_grads is not None and prev_bias_grads is not None:
                prev_weight_errs = np.array(prev_weight_grads[i])
                weights -= momentum * learning_rate * prev_weight_errs
                prev_bias_errs = np.array(prev_bias_grads[i])
                biases -= momentum * learning_rate * prev_bias_errs

            for j, neuron in enumerate(layer):
                neuron.update(input_values=None, weights=weights[j], bias=biases[j])

    def epoch(self, train_data, prev_weight_grads, prev_bias_grads, learning_rate, stop_err, momentum, shuffle_samples):
        if shuffle_samples:
            random.shuffle(train_data)

        global_err = 0
        correct_predictions = 0
        for i, sample in enumerate(train_data):
            result = self.feedforward(sample[0])
            if result.argmax() == np.array(sample[1]).argmax():
                correct_predictions += 1

            weight_grads, bias_grads = self.backpropagation(sample[1])
            global_err += self.calculate_total_error(sample[1])
            self.adjust(weight_grads, prev_weight_grads, bias_grads, prev_bias_grads, learning_rate, momentum)
            prev_weight_grads, prev_bias_grads = weight_grads, bias_grads

        accuracy = correct_predictions / len(train_data)
        fu.save_accuracy(accuracy)

        if stop_err is not None:
            return global_err, prev_weight_grads, prev_bias_grads, global_err < stop_err
        else:
            return global_err, prev_weight_grads, prev_bias_grads

    def train(self, train_data, learning_rate, epochs=None, stop_err=None, momentum=0.0, shuffle_samples=True):
        prev_weight_grads = None
        prev_bias_grads = None

        if epochs is None and stop_err is None:
            raise ValueError("Either epochs or stop_err must be specified")

        fu.clear_errors()
        fu.clear_accuracies()
        if epochs is not None and stop_err is None:
            for epoch in range(epochs):
                global_err, weight_grads, bias_grads = self.epoch(
                    train_data, prev_weight_grads, prev_bias_grads, learning_rate, stop_err, momentum, shuffle_samples
                )
                fu.save_error(global_err)
                prev_weight_grads, prev_bias_grads = weight_grads, bias_grads

        else:
            epoch = 0
            while True:
                global_err, weight_grads, bias_grads, stop_err_reached = self.epoch(
                    train_data, prev_weight_grads, prev_bias_grads, learning_rate, stop_err, momentum, shuffle_samples
                )
                fu.save_error(global_err)
                if stop_err_reached:
                    return

                prev_weight_grads, prev_bias_grads = weight_grads, bias_grads
                epoch += 1
                if epoch == epochs:
                    return
