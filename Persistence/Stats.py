import csv
import matplotlib.pyplot as plt
from scipy import stats

# Test
def linear_regression(x_values, y_values):
    """Returns two coordinates for a linear regression line"""
    regression = stats.linregress(x_values, y_values)

    slope = regression[0]
    intercept = regression[1]

    x_0 = min(x_values)
    x_1 = max(x_values)

    y_0 = intercept
    y_1 = slope * (x_1 - x_0) + y_0

    return [[x_0, x_1], [y_0, y_1]]


def scatter_with_regression_line(x_values, y_values, x_label, y_label):
    """Creates a scatter chart with a linear regression line"""
    regression_line = linear_regression(x_values, y_values)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x_values, y_values)
    plt.plot(regression_line[0], regression_line[1], c='r')
    plt.show()


with open('movie_metadata.csv', newline='') as csv_file:
    metadata_reader = csv.reader(csv_file)
    next(metadata_reader)

    budget = []
    ratings = []

    for row in metadata_reader:
        if row[22] == "":
            continue

        if float(row[22]) > 500000000:
            continue

        budget.append(float(row[22]))
        ratings.append(float(row[25]))

scatter_with_regression_line(budget, ratings, "Movie Budget", "IMDB Rating")
