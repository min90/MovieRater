from pip._vendor.distlib.compat import raw_input


class movierater_input:

    def choose_algorithm(self):
        algoritms = {1: "KNN", 2: "ANN", 3: "PCA"}
        print("--------------------------")
        print("Welcome to the movie rater, where you can see into future and see upcoming movies imdb ratings!")
        print("First choose which algoritm, you would to like to predict with?")
        print("KNN - K-nearest neighbor enter 1")
        print("ANN - Artifical Neural Network enter 2")
        print("PCA - Principal Component Analysis enter 3")
        algorithm_chosen = self.algorithm_run()
        print("You chosed %s" % algoritms[algorithm_chosen])


    def algorithm_run(self):
        algoritm_chosen = 0
        while True:
            try:
                algoritm_chosen = int(raw_input("Enter the number of the algorithm you would like to predict with:"))
            except ValueError:
                print("Invalid input")
            else:
                if algoritm_chosen not in range(1, 4):
                    print("You need to specify an integer in the range of 1 to 3, try again")
                else:
                    return algoritm_chosen



m = movierater_input()
m.choose_algorithm()