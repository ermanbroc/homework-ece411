# 
# Homework 7
#
# Mason Erman
# 
# ECE 411
#

import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma2 = 1
rho = 0.8
d = 4
n = 500

# Covariance matrices
cov1 = sigma2 * np.array([[1, rho],
                          [rho, 1]])
cov2 = sigma2 * np.array([[1, 0],
                          [0, 1]])

# Means
mean1 = np.array([0, 0])
mean2 = np.array([d, 0])

# Simulate data
class1 = np.random.multivariate_normal(mean1, cov1, n)
class2 = np.random.multivariate_normal(mean2, cov2, n)

# Decision function parameters
c = rho / (1 - rho**2)

def f(x1, x2):
    return (
        rho*c*(x1**2 + x2**2)
        - 2*c*x1*x2
        + 2*d*x1
        - d**2
        + sigma2*np.log(1 - rho**2)
    )

# Generate grid over the plot region
x_min = min(class1[:,0].min(), class2[:,0].min()) - 1
x_max = max(class1[:,0].max(), class2[:,0].max()) + 1
y_min = min(class1[:,1].min(), class2[:,1].min()) - 1
y_max = max(class1[:,1].max(), class2[:,1].max()) + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 80),
    np.linspace(y_min, y_max, 80)
)

# All grid points predicted as Class 1
mask_c1 = f(xx, yy) < 0

# -------- PLOT --------
plt.figure(figsize=(8, 8))

# Class 1 samples
plt.plot(class1[:,0], class1[:,1], 'o', label='Class 1')

# Class 2 samples
plt.plot(class2[:,0], class2[:,1], '*', label='Class 2')

# Theoretical Class-1 region (grid points)
plt.plot(xx[mask_c1], yy[mask_c1], '+k', markersize=5, label='Theoretical region C1')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Theoretical Decision Region for Class 1")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("SimulatedDataShaded.png")
