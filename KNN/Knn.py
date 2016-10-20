import Persistence.Reader as rd
import numpy as np
import math
import operator
import random

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
            for y in range(8):
                print(data[x][y])
            if random.random() < split:
                training_set.append(data[x])
            else:
                test_set.append(data[x])
        return training_set, test_set

    def getRangeAccuracy(self, test_point, predictions):
        upper_limit = test_point + 0.5
        lower_limit = test_point - 0.5
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

    def getNeighbors(self, training_set, test_point, k):
        distances = []
        length = len(test_point) - 1
        for x in range(len(training_set)):
            dir_distance = self.euclideanDistance(test_point[0], self.getVector(self.getting_director(training_set[x][0]), "dir"), length)
            actor_distance = self.euclideanDistance(test_point[1], self.getVector(self.getting_actor(training_set[x][1]), "actor"), length)
            distance = dir_distance + actor_distance
            distances.append((training_set[x], distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def average_neighbors_imdb_rating(self, response):
        ratings = 0
        for attr in response:
            ratings += float(attr[2])
        return ratings / len(response)

    # testing
    def test(self):

        test_point = (self.getVector(13, "dir"), self.getVector(432, "actor"))
        #print(self.getNeighbors(self.getData(self.data), test_point, 4))
        #print(self.average_neighbors_imdb_rating(self.getNeighbors(self.getData(self.data), test_point, 4)))
        r = range(0,11,1)
        print(r)

knn = Knn()
knn.test()