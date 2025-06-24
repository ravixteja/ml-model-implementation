import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Number of samples per class
n_samples = 100

# Generate class 1 (inner circle)
r1 = 1.0
theta1 = 2 * np.pi * np.random.rand(n_samples)
x1 = r1 * np.cos(theta1) + 0.1 * np.random.randn(n_samples)
y1 = r1 * np.sin(theta1) + 0.1 * np.random.randn(n_samples)
label1 = np.ones(n_samples)

# Generate class -1 (outer ring)
r2 = 2.5
theta2 = 2 * np.pi * np.random.rand(n_samples)
x2 = r2 * np.cos(theta2) + 0.2 * np.random.randn(n_samples)
y2 = r2 * np.sin(theta2) + 0.2 * np.random.randn(n_samples)
label2 = -1 * np.ones(n_samples)

# Combine the data
X = np.concatenate([np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1)], axis=0)
y = np.concatenate([label1, label2])

# Put into a DataFrame
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y.astype(int)

# Save to CSV for later use
df.to_csv("svm-kernel-dataset.csv", index=False)

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x1, y1, marker='+', label='Class +1')
plt.scatter(x2, y2, marker='o', label='Class -1')
plt.title('Non-Linearly Separable Data for Kernel SVM')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()