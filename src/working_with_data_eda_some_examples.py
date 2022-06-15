import math
import matplotlib.pyplot as plt
import random
from typing import List, Dict
from collections import Counter
from basics.probability import inverse_normal_cdf
from basics.stats import standard_deviation, mean, correlation

random.seed(0)

def random_normal() -> float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())


def random_row() -> List[float]:
    row = [0.0, 0, 0, 0]
    row[0] = random_normal()
    row[1] = -5 * row[0] + random_normal()
    row[2] = row[0] + row[1] + 5 * random_normal()
    row[3] = 6 if row[2] > -2 else 0
    return row


def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)


def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)


def main():
    # uniform between -100 and 100
    uniform = [200 * random.random() - 100 for _ in range(10000)]

    # normal distribution with mean 0, standard deviation 57
    normal = [57 * random_normal()
            for _ in range(10000)]


    print(f"uniform std: {standard_deviation(uniform)}")
    print(f"uniform mean: {mean(uniform)}")
    print(f"normal std: {standard_deviation(normal)}")
    print(f"normal mean: {mean(normal)}")


    # histograms for uniform and normal 
    plot_histogram(uniform, 10, "Uniform Histogram")
    # plt.show()
    plot_histogram(normal, 10, "Normal Histogram")
    # plt.show()

    xs = [random_normal() for _ in range(1000)]
    ys1 = [ x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]


    print(f"xs and ys1 correlation: {correlation(xs, ys1)}") # about 0.9
    print(f"xs and ys2 correlation: {correlation(xs, ys2)}") # about -0.9

    # scatter plot for xs, ys1 and xs, ys2
    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.title("Very Different Joint Distributions")
    # plt.show()


    num_points = 100

    # each row has 4 points, but really we want the columns
    corr_rows = [random_row() for _ in range(num_points)]
    corr_data = [list(col) for col in zip(*corr_rows)]

    # corr_data is a list of four 100-d vectors
    num_vectors = len(corr_data)
    fig, ax = plt.subplots(num_vectors, num_vectors)

    for i in range(num_vectors):
        for j in range(num_vectors):

            # Scatter column_j on the x-axis vs column_i on the y-axis,
            if i != j: 
                ax[i][j].scatter(corr_data[j], corr_data[i])

            # unless i == j, in which case show the series name.
            else: 
                ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                xycoords='axes fraction',
                                ha="center", va="center")

            # Then hide axis labels except left and bottom charts
            if i < num_vectors - 1: 
                ax[i][j].xaxis.set_visible(False)
            if j > 0: 
                ax[i][j].yaxis.set_visible(False)

    # Fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    # plt.show()


if __name__ == "__main__":
   main()