# Homework 6
#
# Mason Erman
# 
# ECE 411

import numpy as np
import matplotlib.pyplot as plt

# Define the true regression function
def f(x):
    return x**2 + 2*x + 1

# Set random seed for reproducibility
np.random.seed(42)

# Generate 50 samples of X from Uniform(-1, 1)
X = np.random.uniform(-1, 1, 50)

# Generate noise e ~ N(0, 1)
e = np.random.normal(0, 1, 50)

# Generate Y using the model Y = f(X) + e
Y = f(X) + e

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='black', marker='o', label='points')

# Plot the true regression function
x_curve = np.linspace(-1, 1, 200)
y_curve = f(x_curve)
plt.plot(x_curve, y_curve, color='black', linestyle='-', linewidth=2, label='actual regression')

delta = 2  # neighborhood radius
x_est = np.arange(-0.9, 0.91, 0.01)  # grid for estimated function
f_hat = []  # store estimated values

prev_val = None  # to handle empty neighborhoods

for x0 in x_est:
    # Find all xi within the radius delta
    idx = np.where(np.abs(X - x0) <= delta)[0]
    
    if len(idx) > 0:
        # Average the corresponding yi
        val = np.mean(Y[idx])
        prev_val = val
    else:
        # If no points in neighborhood, use previous estimate
        if prev_val is not None:
            val = prev_val
        else:
            val = 0  # for the very first point if no neighbors
    f_hat.append(val)

f_hat = np.array(f_hat)

# Plot the estimated regression function (red curve)
plt.plot(x_est, f_hat, color='red', linestyle='-', linewidth=2, label='estimated regression')

# Labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Delta = 2')
plt.legend()
plt.grid(True)
plt.savefig("regression_plot.png")
