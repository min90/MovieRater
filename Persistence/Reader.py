import csv

class CSVReader:

    @staticmethod
    def read(file_path):
        data = []
        nb_directors = 0
        nb_actors = 0

        with open(file_path, 'r', encoding="utf8") as newfile:
            reader = csv.reader(newfile)

            header = True
            for row in reader:

                if not header:
                    data.append(row)

                    # id_dir = int(row[4])
                    # id_1 = int(row[5])
                    # id_2 = int(row[6])
                    # id_3 = int(row[7])
                    #
                    #
                    # if id_dir > nb_directors:
                    #     nb_directors = id_dir
                    # if max(id_1, id_2, id_3) > nb_actors:
                    #     nb_actors = max(id_1, id_2, id_3)

                header = False

        return data #, (nb_directors+1), (nb_actors+1)



    def normalize(self):
        data = self.read("../cleaned_data.csv")
        print("time")
        likes = []
        for like in data:
            likes.append(like[5])
            likes.append(like[6])
            likes.append(like[7])
            likes.append(like[8])
        print("time 2")
        l = self.getXYNormalizedValues(likes)
        data_chunks = [l[x:x + 4] for x in range(0, len(l), 4)]
        for idx, ll in enumerate(data_chunks):
            data[idx][5] = ll[0]
            data[idx][6] = ll[1]
            data[idx][7] = ll[2]
            data[idx][8] = ll[3]
        return data

    def getXYNormalizedValues(self, set):
        xvalues = []
        for x in set:
            xvalues.append(int(x))
        print("time 3")
        normalizedX = self.normalizeData(xvalues)
        print("time 4")
        return normalizedX

    # To normalize data, we have too first do min-max on x values and then on y values.
    # Max(point) = den største x værdi i sættet / Samme med y
    # Min(point) = den mindste x værdi i sættet / Samme med y

    def normalizeData(self, point):
        """Param: point (List) of values"""
        normalized = []
        for p in point:
            xnew = ((p - min(point)) / (max(point) - min(point)))
            normalized.append(xnew)
        return normalized