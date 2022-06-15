from typing import List, Tuple
from basics.linear_algebra import Vector, vector_mean
from basics.stats import standard_deviation


def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """returns the means and standard deviations for each position"""
    dim = len(data[0])

    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]

    return means, stdevs


def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position
    as is if its standard deviation is 0.)
    """
    dim = len(data[0])
    means, stdevs = scale(data)

    # Make a copy of each vector
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled


def main():
    vectors = [[-3., -1, 1], [-1., 0, 1], [1., 1, 1]]
    means, stdevs = scale(vectors)
    assert means == [-1, 0, 1]
    assert stdevs == [2, 1, 0]

    means, stdevs = scale(rescale(vectors))
    assert means == [0, 0, 1]
    assert stdevs == [1, 1, 0]


if __name__ == "__main__":
   main()