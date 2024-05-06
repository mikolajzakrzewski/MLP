from ucimlrepo import fetch_ucirepo
import numpy as np
import neural_network as nn
import file_utils as fu


def train(neural_network):
    stop_condition = input('Stop condition:\n'
                           '(1) – number of epochs\n'
                           '(2) – total error\n'
                           '(3) – both\n')

    epochs = int(input('  * Number of epochs: ')) if stop_condition == '1' or stop_condition == '3' else None
    stop_err = float(input('  * Total error: ')) if stop_condition == '2' or stop_condition == '3' else None
    learning_rate = float(input('  * Learning rate: '))
    momentum = float(input('  * Momentum: ')) if str(input('- Include momentum? (Y/N): ')) == 'Y' else 0.0
    shuffle = True if str(input('- Shuffle training data? (Y/N): ')) == 'Y' else False
    neural_network.train(
        train_data, learning_rate=learning_rate, epochs=epochs,
        stop_err=stop_err, momentum=momentum, shuffle_samples=shuffle
    )


def test(neural_network):
    for test_input in test_inputs:
        print(neural_network.feedforward(test_input))


# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

input_values = X.to_numpy()

classes = list(y['class'])
classes_outputs = {'Iris-setosa': (1, 0, 0), 'Iris-versicolor': (0, 1, 0), 'Iris-virginica': (0, 0, 1)}
output_values = np.array([classes_outputs[i] for i in classes])

train_inputs = []
train_outputs = []
for i in range(output_values.shape[1]):
    range_start = 50 * i
    range_end = range_start + 15
    train_inputs.extend(input_values[range_start:range_end])
    train_outputs.extend(output_values[range_start:range_end])

train_data = [(train_inputs[i], train_outputs[i]) for i in range(len(train_inputs))]

test_inputs = []
test_outputs = []
for i in range(output_values.shape[1]):
    range_start = 50 * i + 15
    range_end = range_start + 35
    test_inputs.extend(input_values[range_start:range_end])
    test_outputs.extend(output_values[range_start:range_end])

test_data = [(test_inputs[i], test_outputs[i]) for i in range(len(test_inputs))]

if str(input('- Load network from file? (Y/N): ')) == 'N':
    hidden_layers_num = int(input('Number of hidden layers: '))
    hidden_layers_sizes = []
    for i in range(hidden_layers_num):
        hidden_layers_sizes.append(int(input('  * ' + str(i + 1) + ' hidden layer size: ')))

    include_bias = True if str(input('- Include bias? (Y/N): ')) == 'Y' else False
    mlp = nn.NeuralNetwork(
        input_values.shape[1], hidden_layers_num, hidden_layers_sizes, output_values.shape[1], include_bias
    )
else:
    mlp = fu.load_obj(str(input('   * Enter file name: ')))

while True:
    choice = input('What to do with the network:\n(1) – train\n(2) – test\n(3) – exit\n')
    if choice == '1':
        train(mlp)
    elif choice == '2':
        test(mlp)
    elif choice == '3':
        break
    else:
        print(' ! Invalid choice !')

if str(input('- Save network to file? (Y/N): ')) == 'Y':
    fu.save_obj(mlp, str(input('   * Enter file name: ')))
