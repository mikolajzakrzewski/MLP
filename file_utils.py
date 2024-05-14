import pickle
import os.path


def save_obj(obj, filename):
    if not os.path.exists('saved_networks/'):
        os.makedirs('saved_networks/')

    with open(os.path.join('saved_networks/', filename + '.pkl'), 'wb') as file:
        pickle.dump(obj, file)


def load_obj(filename):
    with open(os.path.join('saved_networks/', filename + '.pkl'), 'rb') as file:
        return pickle.load(file)


def save_training_error(total_error):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    with open(os.path.join('stats/', 'training_errors.txt'), 'a') as file:
        file.write(str(total_error) + '\n')


def save_validation_error(total_error):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    with open(os.path.join('stats/', 'validation_errors.txt'), 'a') as file:
        file.write(str(total_error) + '\n')


def clear_errors():
    if not os.path.exists('stats/'):
        pass

    with open(os.path.join('stats/', 'training_errors.txt'), 'w'):
        pass

    with open(os.path.join('stats/', 'validation_errors.txt'), 'w'):
        pass


def save_training_accuracy(accuracy):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    with open(os.path.join('stats/', 'training_accuracies.txt'), 'a') as file:
        file.write(str(accuracy) + '\n')


def save_validation_accuracy(accuracy):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    with open(os.path.join('stats/', 'validation_accuracies.txt'), 'a') as file:
        file.write(str(accuracy) + '\n')


def clear_accuracies():
    if not os.path.exists('stats/'):
        pass

    with open(os.path.join('stats/', 'training_accuracies.txt'), 'w'):
        pass

    with open(os.path.join('stats/', 'validation_accuracies.txt'), 'w'):
        pass


def save_outputs(outputs, expected_outputs):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    with open(os.path.join('stats/', 'outputs.txt'), 'a') as f:
        f.write(str(outputs) + '\n')
        f.write(str(expected_outputs) + '\n')
        f.write('\n')


def clear_outputs():
    if not os.path.exists('stats/'):
        pass

    with open(os.path.join('stats/', 'outputs.txt'), 'w'):
        pass
