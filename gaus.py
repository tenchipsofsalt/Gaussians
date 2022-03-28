import numpy as np
import seaborn as sns
import random


def get_random_seq(b):
    # b is beta, s is current sample
    to_return = np.zeros(n)
    to_return[0] = random.choice([-1, 1])

    prev_seq_cost = 0

    for i in range(1, n):
        costs = prev_seq_cost + np.array([0.5 * np.power(-1 - to_return[i - 1], 2), 0.5 * np.power(1 - to_return[i - 1], 2)])
        next_elem_distribution = np.exp(- b * costs)
        next_elem_distribution = next_elem_distribution / np.sum(next_elem_distribution)
        to_return[i] = -1 if random.random() < next_elem_distribution[0] else 1
        prev_seq_cost = costs[0] if to_return[i] == -1 else costs[1]
    return np.expand_dims(to_return, 1)


# number of RVs
n = 5

r = np.expand_dims(np.random.standard_normal(n), 1)

betas = [0.05, 0.1, 0.2, 0.4, 0.8]

for beta in betas:
    mat = np.matmul(r, r.T)
    print(mat.shape)
    print(mat)

