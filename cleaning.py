import csv
import matplotlib.pyplot as plt

with open('movie_metadata.csv', newline='') as csv_file:
    metadata_reader = csv.reader(csv_file)
    next(metadata_reader)

    budget = []
    ratings = []
    directors = []

    for row in metadata_reader:
        if row[22] == "" or row[25] == "":
            continue

        if float(row[22]) > 500000000:
            continue

        budget.append(float(row[22]))
        ratings.append(float(row[25]))
