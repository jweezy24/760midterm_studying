# CHATGPT GENERATED
import numpy as np

# Generate a sample dataset
np.random.seed(123)
n = 100
a_true = 0.0
b_true = 1.0
D = np.random.uniform(a_true, b_true, n)

# Define the likelihood function for the uniform distribution
def likelihood(params, data):
    a, b = params
    n = len(data)
    if np.all(data >= a) and np.all(data <= b):
        likelihood = 1 / (b - a)**n
    else:
        likelihood = 0
    return likelihood

# Calculate the likelihood function for different parameter values
a_values = np.linspace(-1, 2, 100)
b_values = np.linspace(0, 3, 100)
likelihood_values = np.zeros((len(a_values), len(b_values)))
for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        likelihood_values[i, j] = likelihood((a, b), D)

# Plot the likelihood surface
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(a_values, b_values)
ax.plot_surface(X, Y, likelihood_values)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('likelihood')
plt.show()
