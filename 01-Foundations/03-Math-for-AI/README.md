# 📐 Mathematics for AI

## 🎯 Learning Objectives
- Master essential mathematical concepts for AI/ML
- Understand linear algebra, statistics, and calculus basics
- Apply mathematical concepts to real AI problems
- Build intuition for how math powers AI algorithms

## 🔢 Linear Algebra Essentials

### Vectors and Matrices
```python
import numpy as np

# Vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector operations
dot_product = np.dot(v1, v2)  # 32
magnitude = np.linalg.norm(v1)  # 3.74

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
matrix_mult = np.dot(A, B)  # Matrix multiplication
transpose = A.T  # Transpose
inverse = np.linalg.inv(A)  # Inverse (if exists)
```

### Key Concepts for AI
```python
# Eigenvalues and Eigenvectors (PCA)
eigenvals, eigenvecs = np.linalg.eig(A)

# Matrix decomposition (SVD)
U, s, Vt = np.linalg.svd(A)

# Distance metrics
from scipy.spatial.distance import euclidean, cosine
euclidean_dist = euclidean(v1, v2)
cosine_sim = 1 - cosine(v1, v2)
```

## 📊 Statistics and Probability

### Descriptive Statistics
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(100, 15, 1000)  # Normal distribution

# Central tendency
mean = np.mean(data)
median = np.median(data)
mode = np.argmax(np.bincount(data.astype(int)))

# Variability
variance = np.var(data)
std_dev = np.std(data)
range_val = np.max(data) - np.min(data)

print(f"Mean: {mean:.2f}, Std: {std_dev:.2f}")
```

### Probability Distributions
```python
from scipy import stats

# Normal distribution
normal_dist = stats.norm(loc=0, scale=1)
prob = normal_dist.pdf(1.96)  # Probability density at x=1.96

# Binomial distribution
binomial_dist = stats.binom(n=10, p=0.5)
prob_success = binomial_dist.pmf(7)  # P(X=7)

# Visualization
x = np.linspace(-3, 3, 100)
y = normal_dist.pdf(x)
plt.plot(x, y, label='Standard Normal')
plt.legend()
plt.show()
```

### Bayes' Theorem
```python
# P(A|B) = P(B|A) * P(A) / P(B)

def bayes_theorem(prior_a, likelihood_b_given_a, evidence_b):
    """Calculate posterior probability using Bayes' theorem"""
    posterior = (likelihood_b_given_a * prior_a) / evidence_b
    return posterior

# Example: Medical diagnosis
prior_disease = 0.01  # 1% of population has disease
test_accuracy = 0.95  # Test is 95% accurate
false_positive = 0.05  # 5% false positive rate

# P(Disease|Positive Test)
evidence = (test_accuracy * prior_disease) + (false_positive * (1 - prior_disease))
posterior = bayes_theorem(prior_disease, test_accuracy, evidence)
print(f"Probability of disease given positive test: {posterior:.3f}")
```

## 🧮 Calculus for Optimization

### Derivatives and Gradients
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple function: f(x) = x^2
def f(x):
    return x**2

def df_dx(x):
    return 2*x  # Derivative

# Gradient descent example
def gradient_descent(start_x, learning_rate=0.1, iterations=50):
    x = start_x
    history = [x]
    
    for i in range(iterations):
        gradient = df_dx(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Find minimum of f(x) = x^2
final_x, path = gradient_descent(start_x=5.0)
print(f"Minimum found at x = {final_x:.4f}")

# Visualize
x_vals = np.linspace(-6, 6, 100)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, 'b-', label='f(x) = x²')
plt.plot(path, [f(x) for x in path], 'ro-', label='Gradient Descent Path')
plt.legend()
plt.show()
```

### Multivariable Calculus
```python
# Function: f(x,y) = x^2 + y^2
def f_2d(x, y):
    return x**2 + y**2

# Partial derivatives
def df_dx_2d(x, y):
    return 2*x

def df_dy_2d(x, y):
    return 2*y

# Gradient vector
def gradient_2d(x, y):
    return np.array([df_dx_2d(x, y), df_dy_2d(x, y)])

# 2D gradient descent
def gradient_descent_2d(start_point, learning_rate=0.1, iterations=50):
    point = np.array(start_point)
    history = [point.copy()]
    
    for i in range(iterations):
        grad = gradient_2d(point[0], point[1])
        point = point - learning_rate * grad
        history.append(point.copy())
    
    return point, history

# Find minimum
final_point, path_2d = gradient_descent_2d([3.0, 4.0])
print(f"Minimum found at ({final_point[0]:.4f}, {final_point[1]:.4f})")
```

## 🎲 Information Theory

### Entropy and Information Gain
```python
import numpy as np
from collections import Counter

def entropy(labels):
    """Calculate entropy of a dataset"""
    if len(labels) == 0:
        return 0
    
    counts = Counter(labels)
    probabilities = [count/len(labels) for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

def information_gain(parent, left_child, right_child):
    """Calculate information gain from a split"""
    n = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)
    
    parent_entropy = entropy(parent)
    weighted_child_entropy = (n_left/n) * entropy(left_child) + (n_right/n) * entropy(right_child)
    
    return parent_entropy - weighted_child_entropy

# Example: Decision tree split
parent_labels = ['A', 'A', 'B', 'B', 'B', 'C']
left_labels = ['A', 'A', 'B']
right_labels = ['B', 'B', 'C']

gain = information_gain(parent_labels, left_labels, right_labels)
print(f"Information gain: {gain:.3f}")
```

## 🧪 Practical Applications

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Iris Dataset')
plt.show()
```

### Loss Functions and Optimization
```python
# Mean Squared Error
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    # Avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

# Example
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
print(f"Softmax probabilities: {probabilities}")
```

## 🧪 Hands-On Exercises

### Exercise 1: Linear Regression from Scratch
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        for i in range(self.iterations):
            # Forward pass
            y_pred = X.dot(self.weights)
            
            # Calculate loss
            loss = np.mean((y - y_pred)**2)
            
            # Calculate gradients
            gradients = (2/len(y)) * X.T.dot(y_pred - y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.weights)

# Test with synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.1

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

plt.scatter(X, y, alpha=0.6)
plt.plot(X, predictions, 'r-', linewidth=2)
plt.title('Linear Regression from Scratch')
plt.show()
```

### Exercise 2: Logistic Regression
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        for i in range(self.iterations):
            z = X.dot(self.weights)
            predictions = self.sigmoid(z)
            
            # Cross-entropy loss
            loss = -np.mean(y * np.log(predictions + 1e-15) + 
                           (1 - y) * np.log(1 - predictions + 1e-15))
            
            # Gradients
            gradients = X.T.dot(predictions - y) / len(y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X.dot(self.weights))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Test with classification data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

model = LogisticRegression()
model.fit(X, y)
```

## 🎯 Key Takeaways

### Essential Concepts
- **Linear Algebra**: Vectors, matrices, eigenvalues for dimensionality reduction
- **Statistics**: Probability distributions, Bayes' theorem for uncertainty
- **Calculus**: Derivatives and gradients for optimization
- **Information Theory**: Entropy for decision trees and feature selection

### Why Math Matters in AI
- **Optimization**: Gradient descent minimizes loss functions
- **Probability**: Uncertainty quantification and Bayesian inference
- **Linear Algebra**: Efficient computation and dimensionality reduction
- **Statistics**: Data understanding and model validation

### Practical Applications
- **PCA**: Dimensionality reduction using eigendecomposition
- **Gradient Descent**: Core optimization algorithm
- **Loss Functions**: Mathematical objectives for learning
- **Information Gain**: Feature selection in decision trees

---
*Continue to [Data Preprocessing](../04-Data-Preprocessing/) →*