import pickle
import os.path

import numpy as np


def save_obj(obj, filename):
    if not os.path.exists('saved_networks/'):
        os.makedirs('saved_networks/')

    with open(os.path.join('saved_networks/', filename + '.pkl'), 'wb') as file:
        pickle.dump(obj, file)


def load_obj(filename):
    with open(os.path.join('saved_networks/', filename + '.pkl'), 'rb') as file:
        return pickle.load(file)


def save_stat(stat, filename):
    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    with open(os.path.join('stats/', filename + '.txt'), 'a') as file:
        file.write(str(stat) + '\n')


def clear_stats(filename):
    if not os.path.exists('stats/'):
        return

    with open(os.path.join('stats/', filename + '.txt'), 'w'):
        pass
