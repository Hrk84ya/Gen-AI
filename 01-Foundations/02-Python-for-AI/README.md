# 🐍 Python for AI

## 🎯 Learning Objectives
- Master essential Python libraries for AI/ML
- Understand data manipulation with NumPy and Pandas
- Create visualizations with Matplotlib
- Build your first ML model with scikit-learn

## 📚 Essential Libraries

### NumPy - Numerical Computing
```python
import numpy as np

# Arrays and operations
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Mathematical operations
result = np.dot(matrix, matrix)  # Matrix multiplication
mean_val = np.mean(arr)          # Statistical operations
```

### Pandas - Data Manipulation
```python
import pandas as pd

# DataFrames
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Data operations
filtered = df[df['age'] > 25]
grouped = df.groupby('age').mean()
```

### Matplotlib - Visualization
```python
import matplotlib.pyplot as plt

# Basic plotting
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.title('Linear Relationship')
plt.show()
```

## 🧪 Hands-On Exercises

### Exercise 1: Data Analysis
```python
# Load and analyze a dataset
import pandas as pd
import numpy as np

# Create sample data
data = {
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

# Basic analysis
print(df.describe())
print(df.corr())
```

### Exercise 2: First ML Model
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare data
X = df[['feature1', 'feature2']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
```

## 🎯 Key Takeaways
- NumPy for numerical operations
- Pandas for data manipulation
- Matplotlib for visualization
- Scikit-learn for machine learning

---
*Continue to [Mathematics for AI](../03-Math-for-AI/) →*