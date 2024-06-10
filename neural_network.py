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

    def adjust(self, weight_grads, prev_weight_changes, bias_grads, prev_bias_changes, learning_rate, momentum):
        weight_changes = []
        bias_changes = []
        for i, layer in enumerate(self.layers):
            weights = np.array([neuron.weights for neuron in layer])
            weight_change = learning_rate * np.array(weight_grads[i])
            biases = np.array([neuron.bias for neuron in layer])
            bias_change = learning_rate * np.array(bias_grads[i])
            if prev_weight_changes is not None and prev_bias_changes is not None:
                weight_change += momentum * np.array(prev_weight_changes[i])
                bias_change += momentum * np.array(prev_bias_changes[i])

            weight_changes.append(weight_change)
            bias_changes.append(bias_change)
            weights -= weight_change
            biases -= bias_change

            for j, neuron in enumerate(layer):
                neuron.update(input_values=None, weights=weights[j], bias=biases[j])

        return weight_changes, bias_changes

    def epoch(self, train_data, valid_data, prev_weight_changes, prev_bias_changes,
              learning_rate, stop_err, momentum, shuffle_samples, save_values):
        if shuffle_samples:
            random.shuffle(train_data)

        train_error = 0
        correct_train_predictions = 0
        for i, sample in enumerate(train_data):
            result = self.feedforward(sample[0])
            if result.argmax() == np.array(sample[1]).argmax():
                correct_train_predictions += 1

            weight_grads, bias_grads = self.backpropagation(sample[1])
            train_error += self.calculate_total_error(sample[1])
            weight_changes, bias_changes = self.adjust(
                weight_grads, prev_weight_changes, bias_grads, prev_bias_changes, learning_rate, momentum
            )
            prev_weight_changes, prev_bias_changes = weight_changes, bias_changes

        if save_values:
            train_accuracy = correct_train_predictions / len(train_data)
            fu.save_stat(train_accuracy, 'training_accuracies')
            fu.save_stat(train_error, 'training_errors')

        if valid_data is not None:
            valid_error = 0
            correct_valid_predictions = 0
            for i, sample in enumerate(valid_data):
                result = self.feedforward(sample[0])
                if result.argmax() == np.array(sample[1]).argmax():
                    correct_valid_predictions += 1

                valid_error += self.calculate_total_error(sample[1])

            if save_values:
                valid_accuracy = correct_valid_predictions / len(valid_data)
                fu.save_stat(valid_accuracy, 'validation_accuracies')
                fu.save_stat(valid_error, 'validation_errors')

        if stop_err is not None:
            return prev_weight_changes, prev_bias_changes, train_error < stop_err
        else:
            return prev_weight_changes, prev_bias_changes

    def train(self, learning_rate, train_data, valid_data=None,
              epochs=None, stop_err=None, momentum=0.0, shuffle_samples=True):
        prev_weight_changes = None
        prev_bias_changes = None

        if epochs is None and stop_err is None:
            raise ValueError("Either epochs or stop_err must be specified")

        fu.clear_stats('training_errors')
        fu.clear_stats('training_accuracies')
        fu.clear_stats('validation_errors')
        fu.clear_stats('validation_accuracies')
        if epochs is not None and stop_err is None:
            for epoch in range(epochs):
                if (epoch + 1) % 10 == 0:
                    save_values = True
                else:
                    save_values = False

                weight_changes, bias_changes = self.epoch(
                    train_data, valid_data, prev_weight_changes, prev_bias_changes,
                    learning_rate, stop_err, momentum, shuffle_samples, save_values
                )
                prev_weight_changes, prev_bias_changes = weight_changes, bias_changes

        else:
            epoch = 1
            while True:
                if epoch == epochs:
                    return

                if epoch % 10 == 0:
                    save_values = True
                else:
                    save_values = False

                weight_changes, bias_changes, stop_err_reached = self.epoch(
                    train_data, valid_data, prev_weight_changes, prev_bias_changes,
                    learning_rate, stop_err, momentum, shuffle_samples, save_values
                )
                if stop_err_reached:
                    return

                prev_weight_changes, prev_bias_changes = weight_changes, bias_changes
                epoch += 1
