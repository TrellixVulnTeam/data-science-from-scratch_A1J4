import csv
import tqdm
from typing import List
from gradient_descent import gradient_step
from basics.linear_algebra import (
    subtract, 
    Vector, 
    vector_mean, 
    magnitude,
    dot,
    scalar_multiply
)

def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenters the data to have mean 0 in every dimension"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]


def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


def directional_variance(data: List[Vector], w: Vector) -> float:
    """
    Returns the variance of x in the direction of w
    """
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)


def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """
    The gradient of directional variance with respect to w
    """
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data)
            for i in range(len(w))]


def first_principal_component(data: List[Vector],
                              n: int = 100,
                              step_size: float = 0.1) -> Vector:
    # Start with a random guess
    guess = [1.0 for _ in data[0]]

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")

    return direction(guess)


def project(v: Vector, w: Vector) -> Vector:
    """return the projection of v onto the direction w"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)


def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts the result from v"""
    return subtract(v, project(v, w))


def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]


def pca(data: List[Vector], num_components: int) -> List[Vector]:
    """ Returns first num_components principal components """
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)

    return components


def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]


def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    """ 
    Transforms high-dimensional data into low-dimensional using extracted principal components 
    """
    return [transform_vector(v, components) for v in data]


pca_data: List[Vector]

with open("data/pca_data.csv") as f:
    reader = csv.reader(f)
    pca_data = [[float(row[0]), float(row[1])] for row in reader]

de_meaned = de_mean(pca_data)
fpc = first_principal_component(de_meaned)

assert 0.923 < fpc[0] < 0.925
assert 0.382 < fpc[1] < 0.384