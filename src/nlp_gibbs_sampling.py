import random
from collections import defaultdict
from typing import Tuple, Dict, List

# Gibbs sampling illustration
# with simple example 

# x is the value of the first die
# y is the sum of two dice

def roll_a_die() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])


def direct_sample() -> Tuple[int, int]:
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2


def random_y_given_x(x: int) -> int:
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()


def random_x_given_y(y: int) -> int:
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be
        # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be
        # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)


def gibbs_sample(num_iters: int = 100) -> Tuple[int, int]:
    x, y = 1, 2 # doesn't really matter
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y


def compare_distributions(num_samples: int = 1000) -> Dict[int, List[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts


def main():
    n = 10000
    counts = compare_distributions(n)
    print(f"sampling {n} times either with gibbs and direct")
    print(f"sample\tgibbs_cnt/direct_cnt")
    for key, value in counts.items():
        print(f"{key}\t{value[0] / value[1]:.2f}")


if __name__ == "__main__":
    main()