import pickle


def save_obj(obj, filename):
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(obj, file)


def load_obj(filename):
    with open(filename + '.pkl', 'rb') as file:
        return pickle.load(file)


def save_error(epoch, total_error):
    with open('global_errors.txt', 'a') as file:
        file.write('- Epoch ' + str(epoch) + '\n')
        file.write(' * Total error: ' + str(total_error) + '\n\n')


def clear_errors():
    with open('global_errors.txt', 'w'):
        pass
