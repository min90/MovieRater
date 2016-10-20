import math
import operator

class KNN_algorithm:

    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(self, training_set, test_point, k):
        distances = []
        length = len(test_point) - 1
        for x in range(len(training_set)):
            distance = self.euclideanDistance(test_point, training_set[x], length)
            distances.append((training_set[x], distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors


    