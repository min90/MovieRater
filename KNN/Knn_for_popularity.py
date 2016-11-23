import Persistence.Reader as rd
import matplotlib.pyplot as plt
import operator
import random
import math

class Knn:

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

    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    # Get the euclidian distance between two points
    def getDistance(self, trainingSetPoint, pointToTest):
        """
        :param trainingSetPoint: point in the training set
        :param pointToTest: point we want to compare with
        :return: distance between the 2 points
        """
        point1 = [trainingSetPoint[5], trainingSetPoint[6], trainingSetPoint[7], trainingSetPoint[8]]
        point2 = [pointToTest[5], pointToTest[6], pointToTest[7], pointToTest[8]]
        return self.euclideanDistance(point1, point2, 4)

    # Get the point's neighbors
    def getNeighbors(self, training_set, test_point, k):
        """
        :param training_set: training set
        :param test_point: point to get neighbors of
        :param k: amount of neighbors to select
        :return: k-nearest neighbors with their distance to the point
        """
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
        """
        Returns the weighted average score of the neighbors of a test point.
        :param neighbors: The neighbors should be the result of KNN get neighbors
        :param point: The point being tested
        :return: The average weighted imdb Rating - This is our guess..
        """
        sum_distances = 0
        sum_weighted_scores = 0

        for neighbor in neighbors:
            distance = self.getDistance(neighbor, point)
            sum_weighted_scores += (float(neighbor[4]) * distance)
            sum_distances += distance
        if sum_distances != 0:
            return sum_weighted_scores / sum_distances
        else:
            return sum_weighted_scores

    def getAccuracy(self, rating1, rating2):
        """
        Returns the accuracy of two ratings between 0 and 1
        :param rating1: Rating 1
        :param rating2: Rarting 2
        :return: A float accuracy
        """
        if(rating1 < rating2):
            return rating1 /rating2
        else:
            return rating2 / rating1


    # testing
    def test(self):
        # training_data, test_data = self.dividing_set(self.data, 0.67)
        k = 30
        i = 4
        reader = rd.CSVReader()
        # Loop for multiple k
        x, y = [], []
        data = reader.normalize()
        for l in range(1, k):
            total_accuracy = 0
            # Loop i times to make sure it is not just a random result
            for j in range(0, i):
                training_data, test_data = self.dividing_set(data, 0.7)
                accuracy = 0
                for point in test_data:
                    avg_weighted_rating = self.average_neighbors_imdb_rating(self.getNeighbors(training_data, point, l), point)
                    accuracy += self.getAccuracy(float(point[4]), avg_weighted_rating)
                total_accuracy += accuracy/len(test_data)
                print("Try number: " + str(j+1) + " k = " + str(l) + ": average precision: " + str(accuracy/len(test_data)))

            x.append(l)
            y.append(total_accuracy/i)

        plt.plot(x, y)
        plt.ylabel("Accuracy")
        plt.xlabel("k")
        plt.show()

knn = Knn()
knn.test()