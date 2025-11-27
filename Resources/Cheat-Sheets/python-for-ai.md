# 🐍 Python for AI Cheat Sheet

## 🎯 Essential Libraries

### NumPy - Numerical Computing
```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random = np.random.randn(3, 3)

# Array operations
arr + 5          # Broadcasting
arr * 2          # Element-wise multiplication
np.dot(a, b)     # Matrix multiplication
arr.reshape(5, 1) # Reshape array
arr.T            # Transpose

# Statistical operations
np.mean(arr)     # Average
np.std(arr)      # Standard deviation
np.max(arr)      # Maximum value
np.argmax(arr)   # Index of maximum
```

### Pandas - Data Manipulation
```python
import pandas as pd

# DataFrame creation
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Data loading
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')

# Data exploration
df.head()           # First 5 rows
df.info()           # Data types and info
df.describe()       # Statistical summary
df.shape            # Dimensions
df.columns          # Column names

# Data selection
df['name']          # Single column
df[['name', 'age']] # Multiple columns
df.iloc[0]          # First row by index
df.loc[df['age'] > 25] # Conditional selection

# Data manipulation
df.dropna()         # Remove missing values
df.fillna(0)        # Fill missing values
df.groupby('age').mean() # Group by and aggregate
df.sort_values('salary') # Sort by column
```

### Matplotlib - Visualization
```python
import matplotlib.pyplot as plt

# Basic plotting
plt.plot(x, y)              # Line plot
plt.scatter(x, y)           # Scatter plot
plt.bar(categories, values) # Bar plot
plt.hist(data, bins=20)     # Histogram

# Customization
plt.title('My Plot')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend(['Series 1', 'Series 2'])
plt.grid(True)

# Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(x, y1)
ax2.scatter(x, y2)

plt.show()
```

### Scikit-learn - Machine Learning
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Prediction and evaluation
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
```

## 🔧 Data Preprocessing

### Handling Missing Data
```python
# Check for missing values
df.isnull().sum()

# Remove rows with missing values
df.dropna()

# Fill missing values
df.fillna(df.mean())  # With mean
df.fillna(method='forward')  # Forward fill
df.fillna(method='backward') # Backward fill

# Interpolation
df.interpolate()
```

### Feature Engineering
```python
# One-hot encoding
pd.get_dummies(df['category'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['feature1', 'feature2']])

# Creating new features
df['feature_ratio'] = df['feature1'] / df['feature2']
df['feature_interaction'] = df['feature1'] * df['feature2']
```

## 🧠 Deep Learning Basics

### TensorFlow/Keras
```python
import tensorflow as tf
from tensorflow import keras

# Simple neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate
model.evaluate(X_test, y_test)
```

### PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training loop
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch['data'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
```

## 📊 Data Visualization

### Seaborn - Statistical Plots
```python
import seaborn as sns

# Distribution plots
sns.histplot(data=df, x='column')
sns.boxplot(data=df, x='category', y='value')
sns.violinplot(data=df, x='category', y='value')

# Relationship plots
sns.scatterplot(data=df, x='x', y='y', hue='category')
sns.lineplot(data=df, x='time', y='value')
sns.regplot(data=df, x='x', y='y')

# Categorical plots
sns.barplot(data=df, x='category', y='value')
sns.countplot(data=df, x='category')

# Matrix plots
sns.heatmap(df.corr(), annot=True)
sns.clustermap(df.corr())
```

### Plotly - Interactive Plots
```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter plot
fig = px.scatter(df, x='x', y='y', color='category', 
                 hover_data=['additional_info'])
fig.show()

# 3D plot
fig = px.scatter_3d(df, x='x', y='y', z='z', color='category')
fig.show()

# Time series
fig = px.line(df, x='date', y='value', title='Time Series')
fig.show()
```

## 🔍 Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

# ROC AUC
auc = roc_auc_score(y_true, y_pred_proba)
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mse)
```

## 🛠️ Useful Code Snippets

### Data Loading and Saving
```python
# CSV
df = pd.read_csv('file.csv')
df.to_csv('output.csv', index=False)

# JSON
df = pd.read_json('file.json')
df.to_json('output.json', orient='records')

# Pickle (for Python objects)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Memory Optimization
```python
# Reduce memory usage
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
    return df
```

### Progress Bars
```python
from tqdm import tqdm

# For loops
for i in tqdm(range(1000)):
    # Your code here
    pass

# Pandas operations
tqdm.pandas()
df.progress_apply(lambda x: x**2)
```

## 🚀 Performance Tips

### NumPy Optimization
```python
# Use vectorized operations instead of loops
# Slow
result = []
for i in range(len(arr)):
    result.append(arr[i] ** 2)

# Fast
result = arr ** 2

# Use appropriate data types
arr = np.array(data, dtype=np.float32)  # Instead of float64
```

### Pandas Optimization
```python
# Use categorical data for repeated strings
df['category'] = df['category'].astype('category')

# Use vectorized operations
df['new_col'] = df['col1'] + df['col2']  # Instead of apply

# Read only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])
```

## 🐛 Common Pitfalls

### Data Leakage
```python
# Wrong: Scaling before splitting
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)

# Correct: Scale after splitting
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit
```

### Memory Issues
```python
# Use generators for large datasets
def data_generator(file_path, chunk_size=1000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

# Process in chunks
for chunk in data_generator('large_file.csv'):
    process_chunk(chunk)
```

---
*Keep this cheat sheet handy for quick reference! 🚀*