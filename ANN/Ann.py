import Persistence.Reader as rd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import random

reader = rd.CSVReader
data, nb_directors, nb_actors = reader.read("../cleaned_data.csv")

# We divide the dataset in training and test sets
def dividing_set(self, data, split):
    """
    :param data: the whole dataset (cleaned)
    :param split: float between 0 and 1, corresponds to the percentage of the training set
    :return: the training and test sets
    """
    training_set = []
    test_set = []
    for x in range(len(data) - 1):
        if random.random() < split:
            training_set.append(data[x])
        else:
            test_set.append(data[x])
    return training_set, test_set

# sigmoid function
def nonlin(x, deriv = False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in range(0, 10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)
