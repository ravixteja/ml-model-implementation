import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 50 points for class 1 (label: +1)
x1_class1 = np.random.normal(loc=2.5, scale=0.4, size=50)
x2_class1 = np.random.normal(loc=2.5, scale=0.4, size=50)
y_class1 = np.ones(50)

# Generate 50 points for class -1 (label: -1)
x1_class2 = np.random.normal(loc=0.5, scale=0.4, size=50)
x2_class2 = np.random.normal(loc=0.5, scale=0.4, size=50)
y_class2 = -1 * np.ones(50)

# Combine the data
x1 = np.concatenate((x1_class1, x1_class2))
x2 = np.concatenate((x2_class1, x2_class2))
y = np.concatenate((y_class1, y_class2))

# Build DataFrame
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
data = data.round(3)

# Save to CSV
data.to_csv('svm-hard-margin-dataset.csv', index=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(data[data['y'] == 1]['x1'], data[data['y'] == 1]['x2'], marker='*', label='Class +1')
plt.scatter(data[data['y'] == -1]['x1'], data[data['y'] == -1]['x2'], marker='o', label='Class -1')
plt.xlabel('$x_1$', fontsize=13)
plt.ylabel('$x_2$', fontsize=13)
plt.title('Linearly Separable Data for Hard Margin SVM', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
