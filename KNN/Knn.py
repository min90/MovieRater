import Persistence.Reader as rd
import numpy as np
import math
import operator
import random
import timeit
from collections import Counter


class Knn:

    def __init__ (self):
        reader = rd.CSVReader
        self.data, self.nb_directors, self.nb_actors = reader.read("../cleaned_data.csv")

    def dividing_set(self, data, split):
        training_set = []
        test_set = []
        for x in range(len(data) - 1):
            if random.random() < split:
                training_set.append(data[x])
            else:
                test_set.append(data[x])
        return training_set, test_set

    def getDistance(self, trainingSetPoint, pointToTest):
        dir_distance = 1 if (trainingSetPoint[4] != pointToTest[4]) else 0
        actor1_distance = 1 if (trainingSetPoint[5] != pointToTest[5]) else 0
        actor2_distance = 1 if (trainingSetPoint[6] != pointToTest[6]) else 0
        actor3_distance = 1 if (trainingSetPoint[7] != pointToTest[7]) else 0
        return dir_distance + actor1_distance + actor2_distance + actor3_distance

    def getNeighbors(self, training_set, test_point, k):
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


    def average_neighbors_imdb_rating(self, neighbors, point):
        sum_distances = 0
        sum_weighted_scores = 0

        for neighbor in neighbors:
            distance = self.getDistance(neighbor, point)

            sum_weighted_scores += (float(neighbor[8]) * distance)
            sum_distances += distance

        return sum_weighted_scores / sum_distances

    # testing
    def test(self):
        training_data, test_data = self.dividing_set(self.data, 0.67)

        for point in test_data:
            print("Predicting point: " + point[0] + ", " + point[1] + ", " + point[2], ", " + point[3] + "...")
            print("Predicted class = " + str(self.average_neighbors_imdb_rating(self.getNeighbors(training_data, point, 3), point)) + "      actual = " + point[8])

knn = Knn()
knn.test()