import numpy as np
import seaborn as sns


def flip_random_seq(s):
    # # b is beta, this function builds a random sequence for the given beta. We notice as beta gets higher we approach
    # # arrays of constant value, as desired.
    # # Sequences are built by randomly building a sequence based on a sequence in the previous dimension based on the
    # # same beta
    # to_return = np.zeros(n)
    #
    # # WLOG starts with one of these
    # to_return[0] = random.choice([-1, 1])
    #
    # prev_seq_cost = 0
    #
    # for i in range(1, n):
    #     costs = prev_seq_cost + np.array([0.5 * np.power(-1 - to_return[i - 1], 2), 0.5 * np.power(1 - to_return[i - 1], 2)])
    #     next_elem_distribution = np.exp(- b * costs)
    #     next_elem_distribution = next_elem_distribution / np.sum(next_elem_distribution)
    #     to_return[i] = -1 if random.random() < next_elem_distribution[0] else 1
    #     prev_seq_cost = costs[0] if to_return[i] == -1 else costs[1]
    # return np.expand_dims(to_return, 1), prev_seq_cost
    random_idx = np.random.randint(s.shape[0])
    s[random_idx] = - s[random_idx]


def get_first_seq():
    return 2 * (np.random.randint(-1, 1, n) + 0.5)


def calc_seq_cost(b, s):
    cost = 0
    for i in range(1, s.shape[0]):
        cost += 1/2 * np.power(s[i] - s[i - 1], 2)
    return np.exp(- b * cost)


# number of RVs
n = 10
# number of samples to draw for each beta
n_samples = 500

r = np.random.standard_normal(n)

betas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]

for beta in betas:
    mat = np.zeros((n, n))
    seq = get_first_seq()
    # print(seq)
    total_cost = 0

    for j in range(n_samples):
        flip_random_seq(seq)
        seq_cost = calc_seq_cost(beta, seq)
        r_s = np.expand_dims(np.multiply(r, seq), 1)
        # print(r_s)
        # print(seq_cost)
        total_cost += seq_cost
        mat += seq_cost * np.matmul(r_s, r_s.T)
        # print(seq_cost * np.matmul(r_s, r_s.T))

    mat /= total_cost

    print(mat)
    print(total_cost)
print(np.matmul(np.expand_dims(r, 1), np.expand_dims(r, 1).T))
