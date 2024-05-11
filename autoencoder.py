import copy
import neural_network as nn
import plotting as pl


def examine_shuffling(train_data):
    autoencoder = nn.NeuralNetwork(input_layer_size=4, hidden_layers_num=1, hidden_layers_sizes=[2],
                                   output_layer_size=4, include_bias=True)
    learning_rate = 0.6
    shuffle = (True, False)
    autoencoders = [copy.deepcopy(autoencoder), copy.deepcopy(autoencoder)]
    for i, autoencoder in enumerate(autoencoders):
        autoencoder.train(learning_rate=learning_rate, train_data=train_data, epochs=1000, shuffle_samples=shuffle[i])
        if shuffle[i]:
            filename = 'autoencoder_shuffle'
        else:
            filename = 'autoencoder_no_shuffle'

        pl.plot_errors(validation=False, filename=filename)
        pl.plot_accuracies(validation=False, filename=filename)
        hidden_layer_outputs = [neuron.output for neuron in autoencoder.layers[0]]
        print('Hidden layer outputs:')
        print(hidden_layer_outputs)


def examine_training(train_data):
    autoencoder = nn.NeuralNetwork(input_layer_size=4, hidden_layers_num=1, hidden_layers_sizes=[2],
                                   output_layer_size=4, include_bias=True)
    train_parameters = [
        (0.9, 0.0),
        (0.6, 0.0),
        (0.2, 0.0),
        (0.9, 0.6),
        (0.2, 0.9)
    ]
    autoencoders = [copy.deepcopy(autoencoder) for _ in range(len(train_parameters))]
    for i in range(len(train_parameters)):
        autoencoders[i].train(learning_rate=train_parameters[i][0], train_data=train_data,
                              momentum=train_parameters[i][1], epochs=1000, shuffle_samples=False)
        filename = 'autoencoder_' + str(i)
        pl.plot_errors(validation=False, filename=filename)
        pl.plot_accuracies(validation=False, filename=filename)


if __name__ == '__main__':
    autoencoder_train_data = [
        ([1, 0, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 0, 1, 0], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [0, 0, 0, 1])
    ]

    while True:
        choice = input('What to do with the network:\n'
                       '(1) – examine shuffling training data\n'
                       '(2) – examine training with different parameters\n'
                       '(3) – exit\n')
        if choice == '1':
            examine_shuffling(autoencoder_train_data)
        elif choice == '2':
            examine_training(autoencoder_train_data)
        elif choice == '3':
            break
        else:
            print(' ! Invalid choice !')
