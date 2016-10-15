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

                    id_dir = int(row[4])
                    id_1 = int(row[5])
                    id_2 = int(row[6])
                    id_3 = int(row[7])

                    if id_dir > nb_directors:
                        nb_directors = id_dir
                    if max(id_1, id_2, id_3) > nb_actors:
                        nb_actors = max(id_1, id_2, id_3)

                header = False

        return data, (nb_directors+1), (nb_actors+1)