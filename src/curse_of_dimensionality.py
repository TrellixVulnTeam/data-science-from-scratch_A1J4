import random
import tqdm
from typing import List

from matplotlib import pyplot as plt

from basics.linear_algebra import Vector, distance


def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]


def plot_min_and_avg(dimensions: List[int], avg_distances: List[float], min_distances: List[float]):
    plt.title("10.000 Random Distances")
    plt.plot(dimensions, avg_distances, color="blue", label="average distance")
    plt.plot(dimensions, min_distances, color="green", label="minimum distance")
    plt.xlabel("# of dimensions")
    plt.legend()
    plt.show()


def plot_min_avg_ratio(dimensions: List[int], min_avg_ratio: List[float]):
    plt.title("Minimum Distance / Average Distance")
    plt.plot(dimensions, min_avg_ratio)
    plt.xlabel("# of dimensions")
    plt.legend()
    plt.show()


def main():
    dimensions = list(range(1, 101))
    
    avg_distances = []
    min_distances = []
    
    random.seed(0)
    for dim in tqdm.tqdm(dimensions, desc="Curse of Dimensionality"):
        distances = random_distances(dim, 10000)      # 10,000 random pairs
        avg_distances.append(sum(distances) / 10000)  # track the average
        min_distances.append(min(distances))          # track the minimum

    min_avg_ratio = [min_dist / avg_dist
                     for min_dist, avg_dist in zip(min_distances, avg_distances)]

    # plot_min_and_avg(dimensions, avg_distances, min_distances)
    plot_min_avg_ratio(dimensions, min_avg_ratio)


if __name__ == "__main__":
   main()