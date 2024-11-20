import pandas as pd
import numpy as np
import scipy.special
import math
import argparse
import os

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

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_file', type=str, help='Input CSV file from res folder')
    args = parser.parse_args()

    res_file = args.input_file
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

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(res_file), 'compo_results')
    os.makedirs(output_dir, exist_ok=True)

    # Save the results to a CSV file
    base_name = os.path.basename(res_file)
    output_file = os.path.join(output_dir, base_name.replace('res', 'compo_res'))
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
