from DataAnalysis.Relationship import Relationship
import matplotlib.pyplot as plt

class DataAnalysisClass:
    """The class used to do data analysis"""

    @staticmethod
    def insertRelationship(array, relationship, number):
        alreadyThere = False
        r_number = 0
        for r in array:
            if r.name == relationship.name:
                alreadyThere = True
            else:
                alreadyThere = False
                r_number = r.number

        if alreadyThere:
            relationship.number = r_number
        else:
            number += 1
            relationship.number = number
        array.append(relationship)
        return number

    @staticmethod
    def getDirectorAllActorsRelation(data):

        # Make data for director and all actors
        directorAllActors = []

        heading = True
        number = 0
        for row in data:
            if not heading:
                relationship = Relationship(number,
                                            str(row[0]) +
                                            "-" + str(row[1]) +
                                            "-" + str(row[2]) +
                                            "-" + str(row[3]),
                                            float(row[4])
                                            )

                alreadyThere = False
                r_number = 0
                for r in directorAllActors:
                    if r.name == relationship.name:
                        alreadyThere = True
                    else:
                        alreadyThere = False
                        r_number = r.number

                if alreadyThere:
                    relationship.number = r_number
                else:
                    number += 1
                    relationship.number = number
                directorAllActors.append(relationship)
            heading = False

        return directorAllActors

    @staticmethod
    def get_director_single_actor_relation(data):
        # Make data for director and all actors
        directorActor = []

        heading = True
        number = 0
        for row in data:
            if not heading:
                relationship = Relationship(number,
                                            str(row[0]) +
                                            "-" + str(row[1]),
                                            float(row[4])
                                            )
                number = DataAnalysisClass.insertRelationship(directorActor, relationship, number)
                relationship = Relationship(number,
                                            str(row[0]) +
                                            "-" + str(row[2]),
                                            float(row[4])
                                            )
                number =DataAnalysisClass.insertRelationship(directorActor, relationship, number)
                relationship = Relationship(number,
                                            str(row[0]) +
                                            "-" + str(row[3]),
                                            float(row[4])
                                            )
                number = DataAnalysisClass.insertRelationship(directorActor, relationship, number)

            heading = False

        return directorActor

    @staticmethod
    def analyze(directors, actors, data):

        # Get relationships
        director_all_actors = DataAnalysisClass.getDirectorAllActorsRelation(data)
        director_actor = DataAnalysisClass.get_director_single_actor_relation(data)
        # Plot em
        X = []
        Y = []

        for relationship in director_all_actors:
            X.append(relationship.number)
            Y.append(relationship.score)
        plt.figure(1)
        plt.title('The scores of a director and all 3 actors')
        plt.scatter(X, Y)

        X = []
        Y = []
        for relationship in director_actor:
            X.append(relationship.number)
            Y.append(relationship.score)

        plt.figure(2)
        plt.title('The scores of a director and one actor')
        plt.scatter(X, Y)
        plt.show()






