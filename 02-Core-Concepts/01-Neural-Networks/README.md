# 🧠 Neural Networks Fundamentals

## 🎯 Learning Objectives
- Understand the structure of neural networks
- Learn about activation functions and backpropagation
- Build neural networks from scratch and with frameworks
- Apply neural networks to real problems

## 🏗️ Neural Network Architecture

### Basic Structure
```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
```

## 🔥 Activation Functions

### Common Activations
```python
import torch.nn.functional as F

# ReLU (most common)
relu_output = F.relu(x)

# Sigmoid (for binary classification)
sigmoid_output = F.sigmoid(x)

# Tanh (centered around 0)
tanh_output = F.tanh(x)

# Softmax (for multi-class)
softmax_output = F.softmax(x, dim=1)
```

## 📈 Training Process

### Complete Training Loop
```python
import torch.optim as optim

# Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## 🧪 Practical Example: MNIST Classification

```python
import torchvision
import torchvision.transforms as transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Train the model
model = SimpleNN(784, 128, 10)
train_model(model, train_loader)
```

## 🎯 Key Concepts
- **Forward Pass**: Data flows through network layers
- **Backpropagation**: Gradients flow backward to update weights
- **Loss Function**: Measures prediction error
- **Optimizer**: Updates weights to minimize loss

---
*Continue to [Deep Learning](../02-Deep-Learning/) →*