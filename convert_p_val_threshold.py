#!/usr/bin/env python3
import numpy as np
import scipy

# === Parameters
# Sample size in 'base' group
n = 5
# P-value threshold to apply to 'base' group
p = 0.3
# Whether the p-values are two-sided or not
two_sided = True
# Sample size in the target group
other_n = 6

# === Code

dof = n - 2
other_dof = other_n - 2

if two_sided:
	p_mod = p / 2
else:
	p_mod = p
t = scipy.stats.t.ppf(p_mod, dof)
r = t / np.sqrt(n - 2 + np.square(t))

other_t = r * np.sqrt((other_n - 2) / (1 - np.square(r)))
other_p = scipy.stats.t.cdf(other_t, other_dof)
if two_sided:
	other_p *= 2

print(f"Equivalent p-value threshold to {p} (n={n}) for the other group (n={other_n}): {other_p}")

