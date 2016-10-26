import Persistence.Reader as rd
import numpy as np
import math
import operator
import random
import timeit
from collections import Counter

class Knn:

    # Constructor
    def __init__ (self):
        reader = rd.CSVReader
        self.data, self.nb_directors, self.nb_actors = reader.read("../cleaned_data.csv")

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

    # Get the euclidian distance between two points
    def getDistance(self, trainingSetPoint, pointToTest):
        """
        :param trainingSetPoint: point in the training set
        :param pointToTest: point we want to compare with
        :return: distance between the 2 points
        """
        dir_distance = 1 if (trainingSetPoint[4] != pointToTest[4]) else 0
        actor1_distance = 1 if (trainingSetPoint[5] != pointToTest[5]) else 0
        actor2_distance = 1 if (trainingSetPoint[6] != pointToTest[6]) else 0
        actor3_distance = 1 if (trainingSetPoint[7] != pointToTest[7]) else 0
        return (dir_distance + actor1_distance + actor2_distance + actor3_distance)

    # Get the point's neighbors
    def getNeighbors(self, training_set, test_point, k):
        """
        :param training_set: training set
        :param test_point: point to get neighbors of
        :param k: amount of neighbors to select
        :return: k-nearest neighbors with their distance to the point
        """
        start = timeit.default_timer()
        distances = []

        for x in range(len(training_set)):
            distance = self.getDistance(training_set[x], test_point)
            distances.append((training_set[x], distance))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []

        for x in range(k):
            neighbors.append(distances[x][0])

        stop = timeit.default_timer()
        print("Get neighbours time: " + str(stop - start))
        return neighbors

    # Not in use
    def average_neighbors_imdb_rating(self, response):
        ratings = 0
        for attr in response:
            ratings += float(attr[8])
        return ratings / len(response)

    # testing
    def test(self):
        training_data, test_data = self.dividing_set(self.data, 0.67)

        for point in test_data:
            print("Predicting point: " + point[0] + ", " + point[1] + ", " + point[2], ", " + point[3] + "...")
            print("Predicted class = " + str(self.getMajorityClass(self.getNeighbors(training_data, point, 3))) + "      actual = " + str(math.floor(float(point[8]))))

knn = Knn()
knn.test()