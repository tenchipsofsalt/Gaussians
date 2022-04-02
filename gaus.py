import math

import numpy as np
import seaborn as sns


def new_random_seq(s):
    proposed_s = np.copy(s)
    random_idx = np.random.randint(s.shape[0])
    proposed_s[random_idx] = - proposed_s[random_idx]
    return proposed_s


def get_first_seq():
    return 2 * (np.random.randint(-1, 1, n) + 0.5)


def calc_seq_cost(s):
    cost = 0
    for i in range(1, s.shape[0]):
        cost += np.power(s[i] - s[i - 1], 2)
    return 1/2 * cost


# number of RVs
n = 5
# number of samples to draw for each beta
n_samples = 5000

r = np.random.standard_normal(n)

betas = [0, 100]
mats = []

for beta in betas:
    mat = np.zeros((n, n))
    seq = get_first_seq()
    # print(seq)

    for j in range(n_samples):
        proposed = new_random_seq(seq)

        # calculate costs
        cost_delta = calc_seq_cost(proposed) - calc_seq_cost(seq)
        # accept with probability relative to relative probabilities on pmf pB:
        # if cost_delta < 0 or np.random.uniform(0, 1) < math.exp((- beta) * cost_delta):
        # print(math.exp((- beta) * cost_delta))
        if np.random.uniform(0, 1) < math.exp((- beta) * cost_delta):
            seq = proposed
        r_s = np.expand_dims(np.multiply(r, seq), 1)
        # print(r_s)
        # print(seq_cost)
        mat += np.matmul(r_s, r_s.T)
        # print(seq_cost * np.matmul(r_s, r_s.T))

    mat /= n_samples

    print(mat)
    mats.append(mat)
