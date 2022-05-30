import math
from typing import Tuple
from p_value import two_sided_p_value

# Suppose you should deside which ad is more attractive.
# You decide to run an experiment by randomly showing site visitors
# one of the two advertisements and tracking how many people click on each one


def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

# P_A is the probability that people click on ad A
# P_B is the probability that people click on ad B
# null hypothesis is that P_A and P_B are identical


def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


z = a_b_test_statistic(1000, 200, 1000, 180)    # -1.14

assert -1.15 < z < -1.13

# p_value is greater than 5%, hence we don't reject H_0
two_sided_p_value(z)                            # 0.254

assert 0.253 < two_sided_p_value(z) < 0.255

# p_value is smaller than 5%, hence we reject H_0
z = a_b_test_statistic(1000, 200, 1000, 150)    # -2.94
two_sided_p_value(z)                            # 0.003
