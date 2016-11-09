
import Persistence.Reader as rd
import matplotlib.pyplot as plt
import numpy as np
import random

reader = rd.CSVReader
data, nb_directors, nb_actors = reader.read("../cleaned_data.csv")

# We divide the dataset in training and test sets
def dividing_set(self, data):
    """
    :param data: the whole dataset (cleaned)
    :param split: float between 0 and 1, corresponds to the percentage of the training set
    :return: the training and test sets
    """
    training_set = []
    validation_set = []
    test_set = []
    for x in range(len(data) - 1):
        if random.random() < 0.70:
            training_set.append(data[x])
        elif 0.70 < random.random() < 0.85:
            validation_set.append([x])
        else:
            test_set.append(data[x])
    return training_set, test_set

# sigmoid function
def nonlin(x, deriv = False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))


# Use backpropagation



