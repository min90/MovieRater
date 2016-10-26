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

    def getData(self, data):
        realdata = []
        for movie in self.data:
            realdata.append((movie[0], movie[1], movie[8]))
            realdata.append((movie[0], movie[2], movie[8]))
            realdata.append((movie[0], movie[3], movie[8]))
        return realdata

    def getDistance(self, id1, id2):
        return 1 if (id1 == id2) else 0

    def getNeighbors(self, training_set, test_point, k):
        start = timeit.default_timer()
        distances = []
        length = len(test_point) - 1
        for x in range(len(training_set)):

            dir_distance = self.euclideanDistance(self.getVector(self.getting_director(test_point[0]), "dir"), self.getVector(self.getting_director(training_set[x][0]), "dir"), length)
            actor1_distance = self.euclideanDistance(self.getVector(self.getting_actor(test_point[1]), "actor"), self.getVector(self.getting_actor(training_set[x][1]), "actor"), length)
            actor2_distance = self.euclideanDistance(self.getVector(self.getting_actor(test_point[2]), "actor"), self.getVector(self.getting_actor(training_set[x][2]), "actor"), length)
            actor3_distance = self.euclideanDistance(self.getVector(self.getting_actor(test_point[3]), "actor"), self.getVector(self.getting_actor(training_set[x][3]), "actor"), length)
            distance = dir_distance + actor1_distance + actor2_distance + actor3_distance
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