import csv
import random

import numpy as np
from sklearn.datasets import load_iris


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(y):
    return y * (1.0 - y)

# Multi Layer Perceptron
class MLP_NeuralNetwork(object):

    def __init__(self, input, hidden, output):
        """ :param input: number of input neurons
            :param hidden: number of hidden neurons
            :param output: number of output neurons
        """

        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # Set up array ofs 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # Create randomized weights
        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.input, self.hidden)

        # Create arrays of 0 for changes

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # Input activations
        for i in range(self.input - 1): # - 1 is to avoid the bias
            self.ai[i] = inputs[i]

        # Hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # Output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ah[:]


    def backPropagate(self, targets, N):
        """
        :param targets: y values
        :param N: learning rate
        :return: updated weights and current error
        """
        if len(targets) != self.output:
            raise ValueError("Wrong number of targets you silly gooses!")

        # Calculate error terms for output
        # the delta tell you which direction to change the weights

        output_deltas = [0.0] * self.output

        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = derivate_sigmoid(self.ao[k]) * error

        # Calculate terms for hidden
        # Delta tells you which direction to change your weights

        hidden_deltas = [0.0] * self.hidden

        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = derivate_sigmoid(self.ah[j]) * error

        # Update the weights connecting hidden to output

        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= N * change + self.co[j][k]
                self.co[j][k] = change

        # Update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= N * change + self.ci[i][j]
                self.ci[i][j] = change

        # Calcuate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error


    def train(self, patterns, iterations = 3000, N = 0.0002):
        # N : Learning rate

        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)

            if i % 500 == 0:
                print('error %-.5f', error)

    def predict(self, X):
        """
        Return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))

        return predictions

    def test(self, patterns):
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))



