import csv

class CSVReader:

    @staticmethod
    def read(file_path):
        data = []

        with open(file_path, 'r', encoding="utf8") as newfile:
            reader = csv.reader(newfile)

            # reading the file
            header = True
            for row in reader:

                if not header:
                    data.append(row)

                header = False

        return data