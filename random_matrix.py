import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("viridis")


# Gaussian Orthogonal Ensemble
def random_matrix_GOE(n):
    mat = np.zeros((n, n))
    for j in range(n):
        for k in range(j + 1, n):
            mat[j][k] = np.random.normal(0, 1.0 / n)
            mat[k][j] = mat[j][k]
        mat[j][j] *= np.sqrt(2)
    return mat


sample_count = 500
n_buckets = 200
x_range = [-0.5, 0.5]
n_range = [20, 40, 60, 80, 100]
fig, axs = plt.subplots(2)
fig.suptitle('PDF and CDF for GOE Random Matrices')

for n in n_range:
    eigenvalues = np.array(n)
    for i in range(sample_count):
        eigenvalues = np.append(eigenvalues, np.linalg.eig(random_matrix_GOE(n))[0])
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
