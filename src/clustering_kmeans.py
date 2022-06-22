import tqdm
import random
import itertools
from typing import List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from basics.linear_algebra import (
    Vector, 
    vector_mean,
    squared_distance
)


def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


def cluster_means(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    # clusters[i] contains the inputs whose assignment is i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # if a cluster is empty, just use a random point
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]


class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k                      # number of clusters
        self.means = None


    def classify(self, input: Vector) -> int:
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))


    def train(self, inputs: List[Vector]) -> None:
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Check how many assignments changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                # Otherwise keep the new assignments, and compute new means
                assignments = new_assignments
                t.set_description(f"changed: {num_changed} / {len(inputs)}")


def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = [clusterer.classify(input) for input in inputs]

    return sum(squared_distance(input, means[cluster])
                for input, cluster in zip(inputs, assignments))


def main():
    random.seed(12)

    inputs: List[List[float]] = [
        [-14,-5],[13,13],[20,23],[-19,-11],
        [-9,-16],[21,27],[-49,15],[26,13],
        [-46,5],[-34,-1],[11,15],[-49,0],
        [-22,-16],[19,28],[-12,-8],[-13,-19],
        [-41,8],[-11,-6],[-25,-9],[-18,-3]
    ]

    # k choosing technic
    # ks = range(1, len(inputs) + 1)
    # errors = [squared_clustering_errors(inputs, k) for k in ks]
    
    # plt.plot(ks, errors)
    # plt.xticks(ks)
    # plt.xlabel("k")
    # plt.ylabel("total squared error")
    # plt.title("Total Error vs. # of Clusters")
    # plt.show()

    print("reducing image colors to 5")
    image_path = r"data/venice.jpg"
    img = mpimg.imread(image_path) / 256  # rescale to between 0 and 1


    # .tolist() converts a numpy array to a Python list
    pixels = [pixel.tolist() for row in img for pixel in row]

    clusterer = KMeans(5)
    clusterer.train(pixels)   # this might take a while

    def recolor(pixel: Vector) -> Vector:
        cluster = clusterer.classify(pixel)        # index of the closest cluster
        return clusterer.means[cluster]            # mean of the closest cluster
    
    new_img = [[recolor(pixel) for pixel in row]   # recolor this row of pixels
               for row in img]                     # for each row in the image

    fig, (ax_before, ax_after) = plt.subplots(1, 2)
    ax_before.imshow(img)
    ax_before.axis("off")
    ax_before.set_title("before")
    ax_after.imshow(new_img)
    ax_after.axis("off")
    ax_after.set_title("after")
    plt.show()


if __name__ == "__main__":
    main()