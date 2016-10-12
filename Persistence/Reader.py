import csv

class CSVReader:

    @staticmethod
    def read(file_path):
        data = []
        directors = []
        actors = []

        with open(file_path, 'r', encoding="utf8") as newfile:
            reader = csv.reader(newfile)

            # reading the file
            header = True
            for row in reader:

                if not header:
                    data.append(row)

                    if row[0] not in directors:
                        directors.append(row[0])

                    if row[1] not in actors:
                        actors.append(row[1])
                    if row[2] not in actors:
                        actors.append(row[2])
                    if row[3] not in actors:
                        actors.append(row[3])

                header = False

        return directors, actors, data

    # In the method below, the "list" is the array of directors or actors
    # The method search for the given name in the list and get the position
    # Then it creates an array (length : directors or actors number) with zeros and a one at the right position
    @staticmethod
    def getVector(list, name):
        vector = []
        length = len(list)
        position = list.index(name)

        for i in range(0, length):
            if i == position:
                vector.append(1)
            else:
                vector.append(0)

        return vector