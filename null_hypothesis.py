import basics.inference as inference

# H_0 hypotheses is that coin is fair, that is p = 0.5
# let's test it n = 1000 times
mu_0, sigma_0 = inference.normal_approximation_to_binomial(1000, 0.5)

# type 1 error ("false positive") - significance
# getting bounds to make only 5% type 1 error
# (469, 531)
lower_bound, upper_bound = inference.normal_two_sided_bounds(0.95, mu_0, sigma_0)

# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = inference.normal_approximation_to_binomial(1000, 0.55)

# a type 2 error means we fail to reject the null hypothesis
# which will happen when X is still in our original interval
type_2_probability = inference.normal_probability_between(lower_bound, upper_bound, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.887


hi = inference.normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)

type_2_probability = inference.normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.936
print(power)