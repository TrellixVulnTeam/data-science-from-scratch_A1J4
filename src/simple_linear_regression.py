import random
import tqdm
from typing import Tuple

from basics.linear_algebra import Vector
from basics.statistics import correlation, standard_deviation, mean, de_mean
from gradient_descent import gradient_step


def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha
    when the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i


def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))


def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def total_sum_of_squares(y: Vector) -> float:
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))


def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))


def least_squares_gradient_descent_fit(x: Vector, 
                                       y: Vector, 
                                       learning_rate: float = 0.00001,
                                       num_epochs: int = 10000) -> Tuple[float, float]:
    random.seed(0)

    alpha, beta = random.random(), random.random()  # choose random value to start

    with tqdm.trange(num_epochs) as t:
        for _ in t:
            # Partial derivative of loss with respect to alpha
            grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                         for x_i, y_i in zip(x, y))

            # Partial derivative of loss with respect to beta
            grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                         for x_i, y_i in zip(x, y))

            # Compute loss to stick in the tqdm description
            loss = sum_of_sqerrors(alpha, beta, x, y)
            t.set_description(f"loss: {loss:.3f}")

            # Finally, update the guess
            alpha, beta = gradient_step([alpha, beta], [grad_a, grad_b], -learning_rate)

    return alpha, beta


def main():
    x = [float(i) for i in range(-100, 110, 10)]
    y = [3 * i - 5 for i in x]
    print(least_squares_gradient_descent_fit(x, y))


if __name__ == "__main__":
    main()

    