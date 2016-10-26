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

    # The method below compute the vector for the director/actor
    def getVector(self, pos, t):
        length = self.nb_actors if (t == "actor") else self.nb_directors
        vector = np.zeros(length)
        vector[pos] = 1
        return vector

    def getting_actor(self, actor_name):
        for movie in self.data:
            if actor_name in movie and actor_name not in movie[0]:
                index = movie.index(actor_name)
                actor_id = movie[index + 4]
                return int(actor_id)
            else:
                continue
        return -1

    def getting_director(self, director_name):
        for movie in self.data:
            if director_name in movie[0]:
                director_id = movie[4]
                return int(director_id)
            else:
                continue
        return -1

    def dividing_set(self, data, split):
        training_set = []
        test_set = []
        for x in range(len(data) - 1):
            if random.random() < split:
                training_set.append(data[x])
            else:
                test_set.append(data[x])
        return training_set, test_set
    # Not in use
    def getRangeAccuracy(self, test_point, predictions):
        upper_limit = float(test_point[8]) + 0.5
        lower_limit = float(test_point[8]) - 0.5
        if upper_limit >= predictions > lower_limit:
            return True
        else:
            return False

    def getData(self, data):
        realdata = []
        for movie in self.data:
            realdata.append((movie[0], movie[1], movie[8]))
            realdata.append((movie[0], movie[2], movie[8]))
            realdata.append((movie[0], movie[3], movie[8]))
        return realdata

    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((int(instance1[x]) - int(instance2[x])), 2)
        return math.sqrt(distance)

    def getMajorityClass(self, neighbours):
        imdb_classes = []
        for neighbor in neighbours:
            imdb_classes.append(math.floor(float(neighbor[8])))

        votes = Counter(imdb_classes)
        winner, _ = votes.most_common(1)[0]
        return winner