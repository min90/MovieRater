import Persistence.Reader as rd
import matplotlib.pyplot as plt
import operator
import random
import timeit


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
        distances = []

        for x in range(len(training_set)):
            distance = self.getDistance(training_set[x], test_point)
            distances.append((training_set[x], distance))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []

        for x in range(k):
            neighbors.append(distances[x][0])
        

        return neighbors


    def average_neighbors_imdb_rating(self, neighbors, point):
        sum_distances = 0
        sum_weighted_scores = 0

        for neighbor in neighbors:
            distance = self.getDistance(neighbor, point)
            sum_weighted_scores += (float(neighbor[8]) * distance)
            sum_distances += distance
        if sum_distances != 0:
            return sum_weighted_scores / sum_distances
        else:
            return sum_weighted_scores


    def getAccuracy(self, rating1, rating2):
        if(rating1 < rating2):
            return rating1 /rating2
        else:
            return rating2 / rating1


    # testing
    def test(self):
        # training_data, test_data = self.dividing_set(self.data, 0.67)
        k = 20
        i = 5
        # Loop for multiple k
        x, y = [], []
        for l in range(1, k):
            total_accuracy = 0
            for j in range(0, i):
                training_data, test_data = self.dividing_set(self.data, 0.70)
                accuracy = 0
                for point in test_data:
                    # print("Predicting point: " + point[0] + ", " + point[1] + ", " + point[2], ", " + point[3] + "...")
                    avg_weighted_rating = self.average_neighbors_imdb_rating(self.getNeighbors(training_data, point, l), point)
                    accuracy += self.getAccuracy(float(point[8]), avg_weighted_rating)
                    # print("Predicted class = " + str(avg_weighted_rating) + "      actual = " + point[8] + "       accuracy = " + str(self.getAccuracy(float(point[8]), avg_weighted_rating)))
                total_accuracy += accuracy/len(test_data)
                print("Try number: "+ str(j+1) + " k = " + str(l) + ": average precision: " + str(accuracy/len(test_data)))

            x.append(l)
            y.append(total_accuracy/i)

        plt.plot(x, y)
        plt.ylabel("Accuracy")
        plt.xlabel("k")
        plt.show()

knn = Knn()
knn.test()