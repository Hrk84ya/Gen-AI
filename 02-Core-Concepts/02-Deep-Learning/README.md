# 🧠 Deep Learning Architectures

## 🎯 Learning Objectives
- Understand deep neural network architectures
- Master backpropagation and gradient descent
- Implement MLPs, CNNs, and RNNs from scratch
- Learn regularization techniques and optimization
- Build practical deep learning models with TensorFlow/PyTorch

## 📚 Table of Contents
1. [Deep Learning Fundamentals](#fundamentals)
2. [Multi-Layer Perceptrons (MLPs)](#mlps)
3. [Convolutional Neural Networks (CNNs)](#cnns)
4. [Recurrent Neural Networks (RNNs)](#rnns)
5. [Regularization Techniques](#regularization)
6. [Optimization Algorithms](#optimization)
7. [Practical Implementation](#implementation)

## 🔬 Deep Learning Fundamentals {#fundamentals}

### What Makes Networks "Deep"?
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Shallow vs Deep Network Comparison
def create_shallow_network():
    """Single hidden layer network"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def create_deep_network():
    """Deep network with multiple hidden layers"""
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Demonstrate representational power
shallow = create_shallow_network()
deep = create_deep_network()

print(f"Shallow network parameters: {shallow.count_params():,}")
print(f"Deep network parameters: {deep.count_params():,}")
```

### Universal Approximation Theorem
```python
# Demonstrate how deep networks can approximate complex functions
def complex_function(x):
    """Complex function to approximate"""
    return np.sin(x) * np.exp(-x/5) + 0.1 * np.cos(10*x)

# Generate training data
x_train = np.linspace(0, 10, 1000).reshape(-1, 1)
y_train = complex_function(x_train.flatten())

# Create networks of different depths
def create_approximator(depth):
    layers = [keras.layers.Dense(50, activation='relu', input_shape=(1,))]
    for _ in range(depth-1):
        layers.append(keras.layers.Dense(50, activation='relu'))
    layers.append(keras.layers.Dense(1))
    
    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='mse')
    return model

# Train networks of different depths
depths = [1, 3, 5]
models = {}

for depth in depths:
    print(f"Training {depth}-layer network...")
    model = create_approximator(depth)
    model.fit(x_train, y_train, epochs=100, verbose=0)
    models[depth] = model

# Visualize approximations
plt.figure(figsize=(15, 5))
x_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_true = complex_function(x_test.flatten())

for i, depth in enumerate(depths):
    plt.subplot(1, 3, i+1)
    y_pred = models[depth].predict(x_test, verbose=0)
    plt.plot(x_test, y_true, 'b-', label='True function', linewidth=2)
    plt.plot(x_test, y_pred, 'r--', label=f'{depth}-layer network', linewidth=2)
    plt.title(f'{depth}-Layer Network Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🏗️ Multi-Layer Perceptrons (MLPs) {#mlps}

### Forward Propagation
```python
class MLP:
    def __init__(self, layer_sizes):
        """
        Initialize MLP with given layer sizes
        layer_sizes: list of integers, e.g., [784, 128, 64, 10]
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]  # Store activations for backprop
        self.z_values = []      # Store pre-activation values
        
        current_input = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_input = self.relu(z)
            self.activations.append(current_input)
        
        # Output layer (softmax)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output = self.softmax(z_output)
        self.activations.append(output)
        
        return output

# Example usage
mlp = MLP([784, 128, 64, 10])
X_sample = np.random.randn(32, 784)  # Batch of 32 samples
output = mlp.forward(X_sample)
print(f"Output shape: {output.shape}")
print(f"Output probabilities sum: {np.sum(output, axis=1)[:5]}")  # Should be ~1.0
```

### Backpropagation Implementation
```python
def backward(self, X, y, learning_rate=0.01):
    """Backpropagation algorithm"""
    m = X.shape[0]  # Batch size
    
    # Convert labels to one-hot encoding
    y_onehot = np.eye(self.layer_sizes[-1])[y]
    
    # Calculate output layer error
    dz = self.activations[-1] - y_onehot
    
    # Initialize gradients
    dw_list = []
    db_list = []
    
    # Backpropagate through layers
    for i in range(len(self.weights) - 1, -1, -1):
        # Calculate gradients
        dw = np.dot(self.activations[i].T, dz) / m
        db = np.mean(dz, axis=0, keepdims=True)
        
        dw_list.insert(0, dw)
        db_list.insert(0, db)
        
        # Calculate error for previous layer (if not input layer)
        if i > 0:
            dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
    
    # Update weights and biases
    for i in range(len(self.weights)):
        self.weights[i] -= learning_rate * dw_list[i]
        self.biases[i] -= learning_rate * db_list[i]

# Add backward method to MLP class
MLP.backward = backward

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss function"""
    m = y_true.shape[0]
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / m

# Training loop example
def train_mlp(mlp, X_train, y_train, epochs=100, batch_size=32):
    """Train MLP using mini-batch gradient descent"""
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            predictions = mlp.forward(X_batch)
            
            # Calculate loss
            y_onehot = np.eye(mlp.layer_sizes[-1])[y_batch]
            loss = cross_entropy_loss(y_onehot, predictions)
            epoch_loss += loss
            
            # Backward pass
            mlp.backward(X_batch, y_batch)
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses
```

## 🔍 Convolutional Neural Networks (CNNs) {#cnns}

### Convolution Operation
```python
def convolution_2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution operation
    """
    # Add padding if specified
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    # Calculate output dimensions
    output_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1]) // stride + 1
    
    # Initialize output
    output = np.zeros((output_height, output_width))
    
    # Perform convolution
    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            # Extract region
            region = image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # Element-wise multiplication and sum
            output[i//stride, j//stride] = np.sum(region * kernel)
    
    return output

# Example: Edge detection
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# Sobel edge detection kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

# Apply convolution
edges_x = convolution_2d(image, sobel_x)
edges_y = convolution_2d(image, sobel_y)
edges_magnitude = np.sqrt(edges_x**2 + edges_y**2)

# Visualize
plt.figure(figsize=(15, 3))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(edges_x, cmap='gray')
plt.title('Horizontal Edges')
plt.subplot(1, 4, 3)
plt.imshow(edges_y, cmap='gray')
plt.title('Vertical Edges')
plt.subplot(1, 4, 4)
plt.imshow(edges_magnitude, cmap='gray')
plt.title('Edge Magnitude')
plt.tight_layout()
plt.show()
```

### CNN Architecture Implementation
```python
class ConvLayer:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters with Xavier initialization
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2.0 / (filter_size * filter_size))
        self.biases = np.zeros(num_filters)
    
    def forward(self, input_data):
        """Forward pass through convolution layer"""
        self.input_data = input_data
        batch_size, input_height, input_width = input_data.shape
        
        # Add padding
        if self.padding > 0:
            padded_input = np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            padded_input = input_data
        
        # Calculate output dimensions
        output_height = (padded_input.shape[1] - self.filter_size) // self.stride + 1
        output_width = (padded_input.shape[2] - self.filter_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        # Perform convolution for each filter
        for f in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    # Extract region
                    start_i = i * self.stride
                    start_j = j * self.stride
                    region = padded_input[:, start_i:start_i+self.filter_size, start_j:start_j+self.filter_size]
                    
                    # Convolution operation
                    output[:, f, i, j] = np.sum(region * self.filters[f], axis=(1, 2)) + self.biases[f]
        
        return output

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, input_data):
        """Forward pass through max pooling layer"""
        self.input_data = input_data
        batch_size, num_channels, input_height, input_width = input_data.shape
        
        # Calculate output dimensions
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, num_channels, output_height, output_width))
        
        # Perform max pooling
        for i in range(output_height):
            for j in range(output_width):
                start_i = i * self.stride
                start_j = j * self.stride
                region = input_data[:, :, start_i:start_i+self.pool_size, start_j:start_j+self.pool_size]
                output[:, :, i, j] = np.max(region, axis=(2, 3))
        
        return output

# Example CNN architecture
def create_cnn():
    """Create a simple CNN for image classification"""
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

cnn = create_cnn()
cnn.summary()
```

## 🔄 Recurrent Neural Networks (RNNs) {#rnns}

### Vanilla RNN Implementation
```python
class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
    
    def forward(self, inputs, h_prev):
        """
        Forward pass through RNN
        inputs: sequence of inputs (seq_len, batch_size, input_size)
        h_prev: previous hidden state (batch_size, hidden_size)
        """
        seq_len, batch_size = inputs.shape[0], inputs.shape[1]
        
        # Store states for backpropagation
        self.inputs = inputs
        self.hidden_states = np.zeros((seq_len + 1, batch_size, self.hidden_size))
        self.hidden_states[0] = h_prev
        
        outputs = np.zeros((seq_len, batch_size, self.output_size))
        
        # Forward through time
        for t in range(seq_len):
            # Hidden state update
            self.hidden_states[t + 1] = np.tanh(
                np.dot(inputs[t], self.Wxh) + 
                np.dot(self.hidden_states[t], self.Whh) + 
                self.bh
            )
            
            # Output
            outputs[t] = np.dot(self.hidden_states[t + 1], self.Why) + self.by
        
        return outputs, self.hidden_states[-1]
    
    def backward(self, doutputs, dh_next):
        """Backpropagation through time"""
        seq_len, batch_size = doutputs.shape[0], doutputs.shape[1]
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh = dh_next
        
        # Backward through time
        for t in reversed(range(seq_len)):
            # Output layer gradients
            dWhy += np.dot(self.hidden_states[t + 1].T, doutputs[t])
            dby += np.sum(doutputs[t], axis=0, keepdims=True)
            
            # Hidden layer gradients
            dh += np.dot(doutputs[t], self.Why.T)
            
            # Gradient through tanh
            dh_raw = (1 - self.hidden_states[t + 1] ** 2) * dh
            
            # Input and recurrent weight gradients
            dWxh += np.dot(self.inputs[t].T, dh_raw)
            dWhh += np.dot(self.hidden_states[t].T, dh_raw)
            dbh += np.sum(dh_raw, axis=0, keepdims=True)
            
            # Gradient for next iteration
            dh = np.dot(dh_raw, self.Whh.T)
        
        return dWxh, dWhh, dWhy, dbh, dby

# Example: Character-level language model
def create_char_rnn_data(text, seq_length):
    """Create training data for character-level RNN"""
    chars = list(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create sequences
    X, y = [], []
    for i in range(len(text) - seq_length):
        sequence = text[i:i + seq_length]
        target = text[i + seq_length]
        X.append([char_to_idx[ch] for ch in sequence])
        y.append(char_to_idx[target])
    
    return np.array(X), np.array(y), char_to_idx, idx_to_char

# Sample text
sample_text = "hello world this is a sample text for rnn training"
X, y, char_to_idx, idx_to_char = create_char_rnn_data(sample_text, seq_length=10)

print(f"Vocabulary size: {len(char_to_idx)}")
print(f"Training sequences: {len(X)}")
print(f"Sample sequence: {X[0]} -> {y[0]}")
```

### LSTM Implementation
```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for gates
        # Forget gate
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        # Candidate values
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, h_prev, c_prev):
        """Forward pass through LSTM cell"""
        # Concatenate input and previous hidden state
        combined = np.hstack([x, h_prev])
        
        # Forget gate
        f = self.sigmoid(np.dot(combined, self.Wf) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(combined, self.Wi) + self.bi)
        
        # Candidate values
        c_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc)
        
        # Update cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = self.sigmoid(np.dot(combined, self.Wo) + self.bo)
        
        # Update hidden state
        h = o * np.tanh(c)
        
        # Store for backpropagation
        self.cache = (x, h_prev, c_prev, f, i, c_tilde, c, o, combined)
        
        return h, c

# Example usage with TensorFlow/Keras
def create_lstm_model(vocab_size, embedding_dim=128, lstm_units=256):
    """Create LSTM model for text generation"""
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim),
        keras.layers.LSTM(lstm_units, return_sequences=True, dropout=0.2),
        keras.layers.LSTM(lstm_units, dropout=0.2),
        keras.layers.Dense(vocab_size, activation='softmax')
    ])
    
    return model

# Text generation example
def generate_text(model, seed_text, char_to_idx, idx_to_char, length=100):
    """Generate text using trained LSTM model"""
    generated = seed_text
    
    for _ in range(length):
        # Prepare input
        x = np.array([[char_to_idx.get(ch, 0) for ch in generated[-10:]]])
        
        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
    
    return generated
```

## 🛡️ Regularization Techniques {#regularization}

### Dropout Implementation
```python
class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, x, training=True):
        if training:
            # Create random mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
            return x * self.mask
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask

# Batch Normalization
class BatchNormalization:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Store for backward pass
            self.cache = (x, x_norm, batch_mean, batch_var)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out

# L1 and L2 Regularization
def l1_regularization(weights, lambda_l1):
    """L1 regularization penalty"""
    return lambda_l1 * np.sum(np.abs(weights))

def l2_regularization(weights, lambda_l2):
    """L2 regularization penalty"""
    return lambda_l2 * np.sum(weights ** 2)

def regularized_loss(predictions, targets, weights, lambda_l1=0.01, lambda_l2=0.01):
    """Loss function with L1 and L2 regularization"""
    # Base loss (cross-entropy)
    base_loss = -np.mean(targets * np.log(predictions + 1e-15))
    
    # Regularization terms
    l1_penalty = sum(l1_regularization(w, lambda_l1) for w in weights)
    l2_penalty = sum(l2_regularization(w, lambda_l2) for w in weights)
    
    return base_loss + l1_penalty + l2_penalty

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
```

## ⚡ Optimization Algorithms {#optimization}

### Advanced Optimizers
```python
class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params, grads):
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

# Learning Rate Scheduling
class LearningRateScheduler:
    def __init__(self, initial_lr=0.01):
        self.initial_lr = initial_lr
    
    def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10):
        """Step decay schedule"""
        return self.initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    def exponential_decay(self, epoch, decay_rate=0.95):
        """Exponential decay schedule"""
        return self.initial_lr * (decay_rate ** epoch)
    
    def cosine_annealing(self, epoch, T_max=100):
        """Cosine annealing schedule"""
        return self.initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2

# Gradient Clipping
def clip_gradients(grads, max_norm=1.0):
    """Clip gradients by norm"""
    total_norm = 0
    for grad in grads.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for key in grads:
            grads[key] *= clip_coef
    
    return grads
```

## 🚀 Practical Implementation {#implementation}

### Complete Training Pipeline
```python
def train_deep_network(model, train_data, val_data, epochs=100, batch_size=32):
    """Complete training pipeline with all techniques"""
    
    # Initialize optimizer and scheduler
    optimizer = Adam(learning_rate=0.001)
    scheduler = LearningRateScheduler(initial_lr=0.001)
    early_stopping = EarlyStopping(patience=10)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    for epoch in range(epochs):
        # Update learning rate
        current_lr = scheduler.exponential_decay(epoch)
        optimizer.learning_rate = current_lr
        
        # Training phase
        model.training = True
        train_loss = 0
        train_correct = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Forward pass
            predictions = model.forward(X_batch)
            loss = cross_entropy_loss(y_batch, predictions)
            
            # Backward pass
            grads = model.backward(y_batch)
            
            # Gradient clipping
            grads = clip_gradients(grads)
            
            # Update parameters
            optimizer.update(model.parameters, grads)
            
            train_loss += loss
            train_correct += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
        
        # Validation phase
        model.training = False
        val_predictions = model.forward(X_val)
        val_loss = cross_entropy_loss(y_val, val_predictions)
        val_correct = np.sum(np.argmax(val_predictions, axis=1) == np.argmax(y_val, axis=1))
        
        # Calculate metrics
        train_loss /= (len(X_train) // batch_size)
        train_acc = train_correct / len(X_train)
        val_acc = val_correct / len(X_val)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    return history

# Model evaluation
def evaluate_model(model, test_data):
    """Evaluate model on test data"""
    X_test, y_test = test_data
    model.training = False
    
    predictions = model.forward(X_test)
    test_loss = cross_entropy_loss(y_test, predictions)
    test_acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(X_test)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return test_loss, test_acc

# Visualization utilities
def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## 🎯 Key Takeaways

1. **Deep networks** can learn complex hierarchical representations
2. **Backpropagation** enables efficient training of deep architectures
3. **CNNs** excel at spatial pattern recognition
4. **RNNs/LSTMs** handle sequential data effectively
5. **Regularization** prevents overfitting in deep models
6. **Advanced optimizers** improve training stability and speed
7. **Proper initialization** and **normalization** are crucial for deep networks

## 📚 Next Steps

- Explore [Transformer Architecture](../04-Transformers/)
- Learn about [Computer Vision](../03-Computer-Vision/) applications
- Study [Natural Language Processing](../04-NLP/) with deep learning
- Practice with [Practical Projects](../../Projects/)

## 🔗 Additional Resources

- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)