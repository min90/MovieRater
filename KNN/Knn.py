import Persistence.Reader as rd

class Knn:

    def __init__ (self):
        reader = rd.CSVReader
        self.directors, self.actors, self.data = reader.read("../cleaned_data.csv")

    # In the method below, the "list" is the array of directors or actors
    # The method search for the given name in the list and get the position
    # Then it creates an array (length : directors or actors number) with zeros and a one at the right position
    def getVector(self, list, name):
        vector = []
        length = len(list)
        try:
            position = list.index(name)
        except ValueError:
            print("Unable to create the array because Director/Actor was not found!")
            return []

        for i in range(0, length):
            if i == position:
                vector.append(1)
            else:
                vector.append(0)

        return vector

    def test(self):
        print(self.directors)
        print(self.getVector(self.directors, "Daniel Hsia"))

knn = Knn()
knn.test()