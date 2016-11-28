import Persistence.Reader as rd
import numpy as np
import math
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(y):
    return y * (1.0 - y)


# Multi Layer Perceptron
class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output, iterations):
        """ :param input: number of input neurons
            :param hidden: number of hidden neurons
            :param output: number of output neurons
        """
        print("Starting ANN.....")
        self.input = input + 1  # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # Set up var for MLP
        self.train_errors = []
        self.test_errors = []
        self.valid_errors = []
        self.iteration = []
        self.epochs = 0
        self.iterations = iterations

        # Set up array ofs 1s for activations
        self.array_inputs = [1.0] * self.input
        self.array_hidden = [1.0] * self.hidden
        self.array_outputs = [1.0] * self.output

        # Create randomized weights
        self.weights_input = np.random.randn(self.input, self.hidden)
        self.weights_output = np.random.randn(self.hidden, self.output)

        # Create arrays of 0 for changes
        self.changes_input = np.zeros((self.input, self.hidden))
        self.changes_output = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # Input activations
        for i in range(self.input - 1):  # - 1 is to avoid the bias
            self.array_inputs[i] = inputs[i]

        # Hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.array_inputs[i] * self.weights_input[i][j]
            self.array_hidden[j] = sigmoid(sum)

        # Output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.array_hidden[j] * self.weights_output[j][k]
            self.array_outputs[k] = sigmoid(sum)

        return self.array_outputs[:]

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
        # Vi sætter arrayet med det antal pladser vi skal have af outputtet.
        output_deltas = [0.0] * self.output

        for k in range(self.output):
            error = -(targets[k] - self.array_outputs[k])
            output_deltas[k] = derivate_sigmoid(self.array_outputs[k]) * error

        # Calculate terms for hidden
        # Delta tells you which direction to change your weights
        hidden_deltas = [0.0] * self.hidden

        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.weights_output[j][k]
            hidden_deltas[j] = derivate_sigmoid(self.array_hidden[j]) * error

        # Update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.array_hidden[j]
                self.weights_output[j][k] -= N * change + self.changes_output[j][k]
                self.changes_output[j][k] = change

        # Update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.array_inputs[i]
                self.weights_input[i][j] -= N * change + self.changes_input[i][j]
                self.changes_input[i][j] = change

        # Calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.array_outputs[k]) ** 2
        return error

    def train(self, patterns, N=0.05):
        # N : Learning rate

        for i in range(self.iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)

            if i % 10 == 0:
                print('error %-.5f' % error)
                self.train_errors.append(error)
                self.iteration.append(i)

    def test(self, patterns):
        """
        Return list of predictions after training algorithm
        """
        predictions = []
        targets = []
        for p in patterns:
            predictions.append(self.feedForward(p[0]))
            targets.append(p[1])
        self.d_confusion_matrix(targets, predictions)
        return targets, predictions

    def validate(self, patterns):
        predictions = []
        for p in patterns:
            predictions.append(self.feedForward(p[0]))
        return predictions

########################################
#
#   Plotting
#
#########################################

    def plot(self):
        plt.plot(self.iteration, self.train_errors, 'r-', linewidth='2.0')
        plt.plot(self.iteration, self.test_errors, 'c-', linewidth='2.0')
        plt.plot(self.iteration, self.valid_errors, 'k-', linewidth='2.0')
        plt.ylabel("MSE")
        plt.xlabel("Iterations")
        plt.show()

    def prepare_data(self, targets, predicts):
        target_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        predict_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for target_index, target in enumerate(targets):
            m = max(target)
            index_t = target.index(m)
            target_match[index_t] += 1
        for predict_index, predict in enumerate(predicts):
            print(predict)
            m = max(predict)
            print(m)
            index_p = predict.index(m)
            predict_match[index_p] += 1
        print(target_match)
        print(predict_match)
        return target_match, predict_match

    def d_confusion_matrix(self, targets, predicted):
        y_tar, y_pred = self.prepare_data(targets, predicted)
        cm = confusion_matrix(y_tar, y_pred)
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion matrix for IMDB ratings prediction")
        plt.colorbar()
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=45)
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.tight_layout()
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.show()
        print("ACC/MIS: " + str(accuracy_score(y_tar, y_pred)))


def add_zeros():
    output = []
    for i in range(0, 10, 1):
        output.append(0)
    return output


def add_vector_to_output(rating):
    output = add_zeros()
    frac, whole = math.modf(rating)
    if frac < 0.5:
        output[int(whole) - 1] = 1
    elif frac > 0.5:
        output[int(whole)] = 1
    return output


def get_data(data):
    patterns = []
    for movie in data:
        x = []
        x.append(float(movie[5]))
        x.append(float(movie[6]))
        x.append(float(movie[7]))
        x.append(float(movie[8]))
        y = add_vector_to_output(float(movie[4]))
        patterns.append([x, y])
    return patterns


def divide_data(patterns):
    random.shuffle(patterns)
    training = []
    validation = []
    test = []
    for movie in patterns:
        rand = random.random()
        if rand < 0.70:
            training.append(movie)
        elif 0.85 > rand > 0.70:
            validation.append(movie)
        else:
            test.append(movie)
    return training, validation, test


reader = rd.CSVReader()

data = reader.normalize()
patterns = get_data(data)

training, validation, test = divide_data(patterns)

mlp = MLP_NeuralNetwork(4, 7, 10, 100)  # Rule of thumb is 1 hidden layer and the mean between input and output for hidden neurons.
mlp.train(training)
mlp.validate(validation)
mlp.test(test)
# mlp.plot()
