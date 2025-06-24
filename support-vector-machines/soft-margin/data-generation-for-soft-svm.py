import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate 50 points for class +1
x1_pos = np.random.normal(loc=2, scale=0.5, size=50)
x2_pos = np.random.normal(loc=2, scale=0.5, size=50)
y_pos = np.ones(50)

# Generate 50 points for class -1
x1_neg = np.random.normal(loc=0, scale=0.5, size=50)
x2_neg = np.random.normal(loc=0, scale=0.5, size=50)
y_neg = -1 * np.ones(50)

# Inject a few misclassified points (noise)
x1_pos[:3] = np.random.normal(loc=0.5, scale=0.2, size=3)
x2_pos[:3] = np.random.normal(loc=0.5, scale=0.2, size=3)

x1_neg[:3] = np.random.normal(loc=2.5, scale=0.2, size=3)
x2_neg[:3] = np.random.normal(loc=2.5, scale=0.2, size=3)

# Combine
x1 = np.concatenate((x1_pos, x1_neg))
x2 = np.concatenate((x2_pos, x2_neg))
y = np.concatenate((y_pos, y_neg))

# Build DataFrame
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
data = data.round(3)

# Save to CSV
data.to_csv('svm-soft-margin-dataset.csv', index=False)

# Plot
plt.scatter(data[data['y'] == 1]['x1'], data[data['y'] == 1]['x2'], marker='+', label='Class +1')
plt.scatter(data[data['y'] == -1]['x1'], data[data['y'] == -1]['x2'], marker='o', label='Class -1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Noisy Data for Soft Margin SVM')
plt.legend()
plt.grid(True)
plt.show()
