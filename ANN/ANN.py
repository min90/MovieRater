from collections import Counter
from functools import partial
import Persistence.Reader as rd
import math, random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def dot(K, L):
    if len(K) != len(L):
      return 0

    return sum(i[0] * i[1] for i in zip(K, L))

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """returns 1 if the perceptron 'fires', 0 if not"""
    return step_function(dot(weights, x) + bias)

def sigmoid(t):
    print(t)
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""
    outputs = []

    for layer in neural_network:

        input_with_bias = input_vector + [1]             # add a bias input
        output = [neuron_output(neuron, input_with_bias) # compute the output
                  for neuron in layer]                   # for this layer
        outputs.append(output)                           # and remember it

        # the input to the next layer is the output of this one
        input_vector = output

    return outputs

def backpropagate(network, input_vector, target):

    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # adjust weights for output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def patch(x, y, hatch, color):
    """return a matplotlib 'patch' object with the specified
    location, crosshatch pattern, and color"""
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)


def show_weights(neuron_idx):
    weights = network[0][neuron_idx]
    abs_weights = [abs(weight) for weight in weights]

    grid = [abs_weights[row:(row+5)] # turn the weights into a 5x5 grid
            for row in range(0,25,5)] # [weights[0:5], ..., weights[20:25]]

    ax = plt.gca() # to use hatching, we'll need the axis

    ax.imshow(grid, # here same as plt.imshow
              cmap=matplotlib.cm.binary, # use white-black color scale
              interpolation='none') # plot blocks as blocks

    # cross-hatch the negative weights
    for i in range(5): # row
        for j in range(5): # column
            if weights[5*i + j] < 0: # row i, column j = weights[5*i + j]
                # add black and white hatches, so visible whether dark or light
                ax.add_patch(patch(j, i, '/', "white"))
                ax.add_patch(patch(j, i, '\\', "black"))
    plt.show()

if __name__ == "__main__":
    ############################################################################
    #
    #           Read file and make inputs and targets
    #
    ############################################################################
    # The method below compute the vector for the director/actor
    def getVector(pos, t, nb_actors, nb_directors):
        length = nb_actors
        vector = np.zeros(length)

        vector[pos] = 1

        return vector


    def get_data(data):
        patterns = []
        for movie in data:
            x = []
            y = []
            x.append(float(movie[5]))
            x.append(float(movie[6]))
            x.append(float(movie[7]))
            x.append(float(movie[8]))
            y.append(float(movie[4]))
            patterns.append([x, y])
        return patterns


    reader = rd.CSVReader()
    data = reader.read("../cleaned_data.csv")
    targets = get_data(data)

    ##############################################################################
    #
    # DO ANN
    #
    ###############################################################################
    #print(inputs)
    #print(targets)

    random.seed(0)   # to get repeatable results
    input_size = 4  # each input is a vector of length 25
    num_hidden = 10   # we'll have 5 neurons in the hidden layer
    output_size = 1  # we need 10 outputs for each input

    # each hidden neuron has one weight per input, plus a bias weight
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # each output neuron has one weight per hidden neuron, plus a bias weight
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    # the network starts out with random weights
    network = [hidden_layer, output_layer]


    # 10,000 iterations seems enough to converge
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    def predict(input):
        return feed_forward(network, input)[-1]

    for i, input in enumerate(inputs):
        outputs = predict(input)

