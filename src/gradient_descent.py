import random
from typing import (
    TypeVar,
    List,
    Iterator,
    Tuple
)
from basics.linear_algebra import (
    Vector,
    add,
    scalar_multiply,
    vector_mean
)

T = TypeVar('T')  # this allows us to type "generic" functions


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual)
    # squared_error = error ** 2         # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad


def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # Start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


def gradient_descent(dataset: List[Tuple],
                     theta: Vector,
                     learning_rate: float,
                     n_epochs: int) -> Vector:
    """ Computes minimum using gradient descent algorithm starting from theta """
    for epoch in range(n_epochs):
        # Compute the mean of the gradients
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in dataset])
        # Take a step in that direction
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    return theta


def minibatch_gradient_descent(dataset: List[Tuple],
                               theta: Vector,
                               learning_rate: float,
                               batch_size: int,
                               n_epochs: int) -> Vector:
    """ Computes minimum using minibatch gradient descent algorithm starting from theta """
    for epoch in range(n_epochs):
        for batch in minibatches(dataset, batch_size=batch_size):
            grad = vector_mean([linear_gradient(x, y, theta)
                               for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    return theta


def stochastic_gradient_descent(dataset: List[Tuple],
                                theta: Vector,
                                learning_rate: float,
                                n_epochs: int) -> Vector:
    """ Computes minimum using stochastic gradient descent algorithm starting from theta """
    for epoch in range(n_epochs):
        for x, y in dataset:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    return theta


# generating data to use it in gradient descent algorithm
# x ranges from -50 to 49, y is always 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
learning_rate = 0.001
# Start with random values for slope and intercept
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]


# theta = gradient_descent(inputs, theta, learning_rate, n_epochs=5000)
theta = minibatch_gradient_descent(inputs, theta, learning_rate, batch_size=20, n_epochs=1000)
# theta = stochastic_gradient_descent(inputs, theta, learning_rate, n_epochs=100)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
