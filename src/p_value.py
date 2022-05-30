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

