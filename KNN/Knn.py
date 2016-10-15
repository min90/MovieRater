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

    # testing
    def test(self):
        print("directors : " + str(self.nb_directors))
        print("actors : " + str(self.nb_actors))
        print(self.getVector(2, "actor"))

knn = Knn()
knn.test()