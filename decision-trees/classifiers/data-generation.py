import pandas as pd
import numpy as np

# Define possible values for each feature
ages = ['y', 'm', 's']                 # young, middle-aged, senior
incomes = ['l', 'm', 'h']              # low, medium, high
students = ['y', 'n']                  # yes, no
credit_ratings = ['f', 'e']           # fair, excellent
buys_computer = ['y', 'n']            # yes, no

# Set seed for reproducibility
np.random.seed(42)

# Generate the dataset
n_samples = 150
data = {
    'age': np.random.choice(ages, n_samples),
    'income': np.random.choice(incomes, n_samples),
    'student': np.random.choice(students, n_samples),
    'credit_rating': np.random.choice(credit_ratings, n_samples),
}

# Simple logic to make target slightly dependent (optional)
def decide_target(age, income, student, credit):
    # Basic rule: students with fair credit are more likely to buy
    if student == 'y' and credit == 'f':
        return np.random.choice(['y', 'n'], p=[0.8, 0.2])
    elif income == 'h' and age == 'y':
        return np.random.choice(['y', 'n'], p=[0.3, 0.7])
    else:
        return np.random.choice(['y', 'n'])

# Apply rule to determine target
data['buys_computer'] = [
    decide_target(data['age'][i], data['income'][i], data['student'][i], data['credit_rating'][i])
    for i in range(n_samples)
]

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("computer-purchase-data.csv", index=False)

# Display first few rows
print(df.head())
