import random
import basics.inference as inference

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from a N(mu, sigma)?
    """
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * inference.normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * inference.normal_probability_below(x, mu, sigma)


# H_0 hypotheses is that coin is fair, that is p = 0.5
# let's test it n = 1000 times
mu_0, sigma_0 = inference.normal_approximation_to_binomial(1000, 0.5)

two_sided_p_value(529.5, mu_0, sigma_0)   # 0.062

upper_p_value = inference.normal_probability_above
lower_p_value = inference.normal_probability_below

upper_p_value(524.5, mu_0, sigma_0) # 0.061 accepting null hypothesis (> 5%)

upper_p_value(526.5, mu_0, sigma_0) # 0.047 rejecting null hypothesis (< 5%)

# experiment for the above mentioned results
extreme_value_count = 0
for _ in range(10000):
    num_heads = sum(1 if random.random() < 0.5 else 0    # Count # of heads
                    for _ in range(1000))                # in 1000 flips,
    if num_heads >= 530 or num_heads <= 470:             # and count how often
        extreme_value_count += 1                         # the # is 'extreme'

# p-value was 0.062 => ~62 extreme values out of 1000
assert 590 < extreme_value_count < 650, f"{extreme_value_count}"
print("experiment succeeded")