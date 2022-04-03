import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_palette("rocket")


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
    return 1 / 2 * cost


# number of RVs
n = 100
# number of samples to draw for each beta
n_samples = 5000
# number of GRVs for each beta
n_grv = 100

r = np.random.standard_normal(n)

betas = [0, 0.01, 0.1, 1, 10, 100]
mats = []

for beta in betas:
    mat = np.zeros((n, n))
    seq = get_first_seq()
    # print(seq)

    for j in range(n_samples):
        proposed = new_random_seq(seq)

        # calculate costs
        cost_delta = calc_seq_cost(proposed) - calc_seq_cost(seq)
        # accept with probability relative to relative probabilities on pmf pB (note if delta is negative this will
        # always accept):
        if np.random.uniform(0, 1) < math.exp((- beta) * cost_delta):
            seq = proposed
        r_s = np.expand_dims(np.multiply(r, seq), 1)
        # print(r_s)
        # print(seq_cost)
        mat += np.matmul(r_s, r_s.T)
        # print(seq_cost * np.matmul(r_s, r_s.T))

    mat /= n_samples
    mats.append(mat)


fig, axs = plt.subplots(1, len(betas))
fig.suptitle("Transition from Diagonal to Full-Rank Matrix for Various Betas")
fig.set_size_inches(12, 4)
for i in range(len(betas)):
    axs[i].imshow(mats[i], interpolation='none', label='beta: ' + str(betas[i]))
    axs[i].axis('off')
    axs[i].set_title('beta: ' + str(betas[i]))
plt.show()

# plot GRVs

fig, axs = plt.subplots(len(betas), 1)
fig.suptitle("Sampled GRVs from Various Betas")
fig.set_size_inches(6, len(betas) * 4 - 5)
mean = np.zeros(n)
x = np.arange(0, n)

for i in range(len(betas)):
    grv = np.swapaxes(np.random.multivariate_normal(mean, mats[i], n_grv), 0, 1)
    axs[i].plot(x, grv, label='beta: ' + str(betas[i]), color=sns.color_palette("viridis")[3], alpha=0.1)
    axs[i].set_ylim((-10, 10))
    axs[i].set_title('beta: ' + str(betas[i]))
plt.show()



