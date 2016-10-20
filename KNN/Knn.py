import random

import Persistence.Reader as rd
import numpy as np

class Knn:

    def __init__ (self):
        reader = rd.CSVReader
        self.data, self.nb_directors, self.nb_actors = reader.read("../cleaned_data.csv")

    # The method below compute the vector for the director/actor
    def getVector(self, pos, type):
        length = self.nb_actors if (type == "actor") else self.nb_directors
        vector = np.zeros(length)
        vector[pos] = 1

        return vector

    def getting_actor(self, actor_name):
        for movie in self.data:
            if actor_name in movie and actor_name not in movie[0]:
                index = movie.index(actor_name)
                actor_id = movie[index + 4]
                return actor_id
            else:
                continue
        return -1

    def getting_director(self, director_name):
        for movie in self.data:
            if director_name in movie[0]:
                director_id = movie[4]
                return director_id
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


    # testing
    def test(self):
        print("directors : " + str(self.nb_directors))
        print("actors : " + str(self.nb_actors))
        print(self.getVector(323, "actor"))
        print(self.data[0][6])
        print(self.getting_actor("Clint Eastwood"))
        print(self.getting_director("Clint Eastwood"))
        print(type(self.getting_actor("Clint Eastwood")))
        print(self.getVector(int(self.getting_actor("Clint Eastwood")), "actor"))
        trainingSet, testSet = self.dividing_set(self.data, 0.67)
        print(trainingSet)
        print(testSet)

knn = Knn()
knn.test()