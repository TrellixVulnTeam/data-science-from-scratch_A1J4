import random
from typing import List, TypeVar, Callable

from basics.stats import median, standard_deviation

X = TypeVar('X')        # Generic type for data
Stat = TypeVar('Stat')  # Generic type for "statistic"


def bootstrap_sample(data: List[X]) -> List[X]:
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data: List[X],
                        stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


def main():
    
    # 101 points all very close to 100
    close_to_100 = [99.5 + random.random() for _ in range(101)]

    # 101 points, 50 of them near 0, 50 of them near 200
    far_from_100 = ([99.5 + random.random()] +
                    [random.random() for _ in range(50)] +
                    [200 + random.random() for _ in range(50)])

    medians_close = bootstrap_statistic(close_to_100, median, 100)
    medians_far = bootstrap_statistic(far_from_100, median, 100)
    
    print(f"std medians_close: {standard_deviation(medians_close)}")
    print(f"std medians_far: {standard_deviation(medians_far)}")


if __name__ == "__main__":
    main()