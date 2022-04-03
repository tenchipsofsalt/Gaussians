import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("viridis")


# Gaussian Orthogonal Ensemble
def random_matrix_GOE(size):
    mat = np.zeros((size, size))
    for j in range(size):
        for k in range(j + 1, size):
            mat[j][k] = np.random.normal(0, 1.0 / size)
            mat[k][j] = mat[j][k]
        mat[j][j] *= np.sqrt(2)
    return mat


sample_count = 1000
n_buckets = 200
x_range = [-0.5, 0.5]
n_range = [20, 40, 60, 80, 100]
fig, axs = plt.subplots(2)
fig.suptitle('Eigenvalue Distribution (PDF/CDF) for GOE Random Matrices')

for n in n_range:
    eigenvalues = []
    for i in range(sample_count):
        for e in np.linalg.eig(random_matrix_GOE(n))[0]:
            eigenvalues.append(e)
    eigenvalues = np.array(eigenvalues)
    buckets = np.zeros(n_buckets)
    for i in range(n_buckets):
        buckets[i] = np.count_nonzero(
            np.logical_and(eigenvalues <= (x_range[0] + (x_range[1] - x_range[0]) * (i + 1) / n_buckets),
                           eigenvalues > (x_range[0] + (x_range[1] - x_range[0]) * i / n_buckets)))
    buckets /= np.sum(buckets)

    X = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / n_buckets)
    axs[0].plot(X, buckets, label=str(n) + 'pdf')
    axs[1].plot(X, np.cumsum(buckets), label=str(n) + 'cdf')
axs[0].legend(loc="upper left")
axs[1].legend(loc="upper left")
plt.show()

# Wishart ensemble
n = 100
m_range = [25, 50, 75, 100, 125, 150]
x_range = [0.001, 5]


def create_wishart_ensemble(n_val, m_val):
    n_by_m = np.random.normal(0, 1, (n_val, m_val))
    mat = np.matmul(n_by_m, n_by_m.T)
    return mat / n_val


fig, axs = plt.subplots(2)
fig.suptitle('Eigenvalue Distribution (PDF/CDF) for Wishart Ensemble')

for m in m_range:
    eigenvalues = []
    for i in range(sample_count):
        for e in np.linalg.eig(create_wishart_ensemble(n, m))[0]:
            # in case this turns out complex for some reason
            eigenvalues.append(np.real(e))
    eigenvalues = np.array(eigenvalues)
    buckets = np.zeros(n_buckets)
    for i in range(n_buckets):
        buckets[i] = np.count_nonzero(
            np.logical_and(eigenvalues <= (x_range[0] + (x_range[1] - x_range[0]) * (i + 1) / n_buckets),
                           eigenvalues > (x_range[0] + (x_range[1] - x_range[0]) * i / n_buckets)))
    buckets /= np.sum(buckets)

    X = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / n_buckets)
    axs[0].plot(X, buckets, label='pdf lambda: ' + str(m / n))
    axs[1].plot(X, np.cumsum(buckets), label='cdf lambda: ' + str(m / n))
axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")
plt.show()
