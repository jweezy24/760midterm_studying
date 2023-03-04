# GENERATED FROM CHATGPT

import numpy as np
from scipy.stats import norm

# Generate a sample dataset
np.random.seed(123)
n = 100
mu_true = 2.0
sigma_true = 1.0
D = np.random.normal(mu_true, sigma_true, n)

# Define the likelihood function for the normal distribution
def likelihood(params, data):
    mu, sigma = params
    n = len(data)
    log_likelihood = -0.5*n*np.log(2*np.pi*sigma**2) - np.sum((data-mu)**2 / (2*sigma**2))
    return np.exp(log_likelihood)

# Calculate the likelihood function for different parameter values
mu_values = np.linspace(-5, 5, 100)
sigma_values = np.linspace(0.1, 5, 100)
likelihood_values = np.zeros((len(mu_values), len(sigma_values)))
for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        likelihood_values[i, j] = likelihood((mu, sigma), D)

# Plot the likelihood surface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(mu_values, sigma_values)
ax.plot_surface(X, Y, likelihood_values)
ax.set_xlabel('mu')
ax.set_ylabel('sigma')
ax.set_zlabel('likelihood')
plt.show()