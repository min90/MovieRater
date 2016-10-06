from DataAnalysis.Relationship import Relationship
import matplotlib.pyplot as plt

class DataAnalysisClass:
    """The class used to do data analysis"""

    def getDirectorAllActorsRelation(self, data):
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

                for r in directorAllActors:
                    if r.name == relationship.name:
                        relationship.number = r.number
                    else:
                        number += 1
                        directorAllActors.append(relationship)
            heading = False

        return directorAllActors

    def analyze(self, data):

        # Get relationships
        directorAllActors = self.getDirectorAllActorsRelation(data)

        #Plot em








