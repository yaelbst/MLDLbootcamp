import numpy as np
import pylab as plt
from collections import namedtuple
from sklearn.datasets import load_iris

# Data preparation
# 1. Load the iris dataset .
iris = load_iris()
X = iris.data

# 2. randomly choose the starting centroids/means as three of the points from datasets
k = 3
n, d = X.shape
mu = X[np.random.choice(n, k, False), :]

# 3. initialize the variances for each gaussians
Sigma = [np.eye(d)] * k

# 4. initialize the probabilities/weights for each gaussians, as equally distributed
w = [1. / k] * k

# 5. Responsibility (membership) matrix is initialized to all zeros
R = np.zeros((n, k))


# Expectation
# 6. Write a P function that calculates for each point the probability of belonging to each gaussian
def prob(s, m):
    Data = X - np.tile(np.transpose(m), (n, 1))
    prob = np.sum(np.dot(Data, np.linalg.inv(s)) * Data, 1)
    prob = np.exp(-0.5 * prob) / np.sqrt((np.power((2 * np.pi), d)) * np.absolute(np.linalg.det(s)))
    return prob

def E_Step(R):
    # 7. Write the E-step (expectation) in which we multiply this P function for every point by the weight of the corresponding cluster

    for i in range(k):
        R[:, i] = w[i] * prob(Sigma[i], mu[i])

    # 9. Normalize the responsibility matrix by the sum
    R = (R.T / np.sum(R, axis=1)).T

    # 8. Sum the log likelihood of all clusters
    log_likelihood = np.sum(np.log(np.sum(R, axis=1)))

    # 10. Calculate the number of points belonging to each Gaussian
    N_ks = np.sum(R, axis=0)
    return log_likelihood, N_ks, R

log_likelihood, N_ks, R = E_Step(R)

print(log_likelihood)

print(R)