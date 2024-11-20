import pandas as pd
import numpy as np
import scipy.special
import math


def binom(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)


def gpi_alpha(order, alpha, i):
    res = alpha
    for j in range(i):
        res = res * order / (order - 1) - 1 / order
    return res


def stepi_compo(alpha, sigma, q, cn, i, order):
    alpha = int(np.ceil(gpi_alpha(order, alpha, int(i))))
    res = []
    for k in range(alpha + 1):
        coeff = np.log(binom(alpha, k) * math.pow(1 - q, alpha - k) * math.pow(q, k))
        expect = math.pow(cn, 2) * k * (k - 1) / (2 * math.pow(sigma, 2))
        res.append(coeff + expect)
    divergence = scipy.special.logsumexp(res) / (alpha - 1)
    return divergence * order * (alpha - 1)


def scale(x, i, order, alpha):
    temp = i * np.log(order - 1) - (i + 1) * np.log(order) + np.log(x) - np.log(alpha - 1)
    return np.exp(temp)


res_file = "res/res.csv"  # make sure to modify this to the correct sensitivity result file
feature = 'distance (sum)'
dataset = "MNIST"
model = "lenet"
cn = 1
eps = 10
alpha = 8

df = pd.read_csv(res_file)

print(f"Using {len(df['step'].unique())} checkpoints")
assert len(df['sigma'].unique()) == 1
assert len(df['batch_size'].unique()) == 1
sigma = df['sigma'].unique()[0]
bs = df['batch_size'].unique()[0]
order = df['step'].max() * 3


if dataset == "MNIST":
    p = bs / 60000
elif dataset == "CIFAR10":
    p = bs / 50000
else:
    raise NotImplementedError

# First compute the scaled per-instance guarantee for composition for each step (before expectation)
df[feature] = df.apply(lambda x: stepi_compo(alpha, sigma, p, x[feature] * bs, x['step'], order), axis=1)



# Now compute the expectation over the trials to obtain the per-step contribution in composition
df = df.groupby(["point", "step"], as_index=False)[feature].apply(lambda grp: scipy.special.logsumexp(grp) - np.log(grp.count()))

df[feature] = df.apply(lambda x: scale(x[feature], x['step'], order, alpha), axis=1)

df = df.rename(columns={feature: "Privacy cost"})

# Output the results in the terminal
print(df)

# Save the results to a CSV file
output_file = "res_compo.csv"
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
