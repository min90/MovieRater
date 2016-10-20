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

    def getting_director(self, director_name):
        for movie in self.data:
            if director_name in movie[0]:
                director_id = movie[4]
                return director_id
            else:
                continue

    # testing
    def test(self):
        print("directors : " + str(self.nb_directors))
        print("actors : " + str(self.nb_actors))
        print(self.getVector(2, "actor"))
        print(self.data[0][6])
        print(self.getting_actor("Clint Eastwood"))
        print(self.getting_director("Clint Eastwood"))

knn = Knn()
knn.test()