# Deep Learning Architectures

## 🎯 Learning Objectives
By the end of this module, you will:
- Master multi-layer perceptrons (MLPs) and deep feedforward networks
- Understand convolutional neural networks (CNNs) for computer vision
- Learn recurrent neural networks (RNNs) for sequential data
- Implement modern architectures like ResNet, LSTM, and Attention
- Apply regularization and optimization techniques
- Build end-to-end deep learning pipelines

## 🧠 What is Deep Learning?

Deep learning uses neural networks with multiple hidden layers to learn hierarchical representations of data. Unlike traditional machine learning, deep learning can automatically discover features from raw data.

### Key Characteristics:
- **Hierarchical Learning**: Each layer learns increasingly complex features
- **End-to-End Training**: Learn features and classifier jointly
- **Scalability**: Performance improves with more data and compute
- **Versatility**: Works across domains (vision, NLP, speech, etc.)

## 📚 Module Contents

### 1. [Multi-Layer Perceptrons (MLPs)](./01_MLP_from_scratch.ipynb)
- Deep feedforward networks
- Universal approximation theorem
- Backpropagation in depth
- Regularization techniques

### 2. [Convolutional Neural Networks](./02_CNN_architectures.ipynb)
- Convolution and pooling operations
- CNN building blocks
- Classic architectures (LeNet, AlexNet, VGG)
- Modern innovations (ResNet, DenseNet)

### 3. [Recurrent Neural Networks](./03_RNN_and_LSTM.ipynb)
- Vanilla RNNs and vanishing gradients
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Bidirectional and stacked RNNs

### 4. [Advanced Architectures](./04_advanced_architectures.ipynb)
- Attention mechanisms
- Transformer building blocks
- Residual connections
- Batch normalization and layer normalization

### 5. [Optimization Techniques](./05_optimization_techniques.ipynb)
- Gradient descent variants (SGD, Adam, RMSprop)
- Learning rate scheduling
- Gradient clipping and normalization
- Advanced optimizers (AdamW, LAMB)

### 6. [Regularization Methods](./06_regularization_methods.ipynb)
- Dropout and its variants
- Weight decay and L1/L2 regularization
- Early stopping and model selection
- Data augmentation strategies

### 7. [Transfer Learning](./07_transfer_learning.ipynb)
- Pre-trained model usage
- Fine-tuning strategies
- Feature extraction vs fine-tuning
- Domain adaptation techniques

## 🏗️ Architecture Evolution

### Timeline of Deep Learning Breakthroughs

```
1986: Backpropagation Algorithm
1989: LeNet (Convolutional Networks)
1997: LSTM (Long Short-Term Memory)
2006: Deep Belief Networks
2012: AlexNet (ImageNet Revolution)
2014: VGG, GoogLeNet
2015: ResNet (Residual Networks)
2017: Transformer (Attention Is All You Need)
2018: BERT (Bidirectional Transformers)
2019: GPT-2 (Large Language Models)
2020: Vision Transformer (ViT)
2021: GPT-3, DALL-E
2022: ChatGPT, Stable Diffusion
```

## 🔧 Essential Concepts

### 1. **Depth vs Width**
- **Depth**: Number of layers (enables hierarchical learning)
- **Width**: Number of neurons per layer (increases capacity)
- **Trade-offs**: Deeper networks can be more expressive but harder to train

### 2. **Activation Functions**
```python
# Modern activation functions
import torch.nn as nn

activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(0.01),
    'ELU': nn.ELU(),
    'Swish': nn.SiLU(),  # Swish/SiLU
    'GELU': nn.GELU(),
    'Mish': nn.Mish()
}
```

### 3. **Normalization Techniques**
```python
# Different normalization methods
normalizations = {
    'BatchNorm': nn.BatchNorm2d(channels),
    'LayerNorm': nn.LayerNorm(features),
    'GroupNorm': nn.GroupNorm(groups, channels),
    'InstanceNorm': nn.InstanceNorm2d(channels)
}
```

### 4. **Loss Functions**
```python
# Common loss functions
losses = {
    'CrossEntropy': nn.CrossEntropyLoss(),
    'MSE': nn.MSELoss(),
    'MAE': nn.L1Loss(),
    'Huber': nn.SmoothL1Loss(),
    'Focal': FocalLoss(),  # For imbalanced data
    'Contrastive': ContrastiveLoss()  # For metric learning
}
```

## 🎯 Architecture Design Principles

### 1. **Convolutional Networks (CNNs)**
```python
class ModernCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 2. **Residual Networks (ResNet)**
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.skip(residual)
        out = F.relu(out)
        
        return out
```

### 3. **Attention Mechanism**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
```

## 📊 Training Strategies

### 1. **Learning Rate Scheduling**
```python
# Common learning rate schedules
schedulers = {
    'StepLR': torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
    'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
    'ReduceOnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10),
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
}
```

### 2. **Advanced Training Techniques**
```python
# Mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. **Model Ensembling**
```python
class ModelEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
```

## 🔍 Model Analysis and Interpretation

### 1. **Feature Visualization**
```python
def visualize_feature_maps(model, input_tensor, layer_name):
    """Visualize intermediate feature maps"""
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(get_activation(layer_name))
    
    # Forward pass
    _ = model(input_tensor)
    
    # Visualize
    feature_maps = activation[layer_name][0]  # First sample
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[0]:
            ax.imshow(feature_maps[i].cpu(), cmap='viridis')
        ax.axis('off')
    plt.show()
```

### 2. **Gradient-based Explanations**
```python
def grad_cam(model, input_tensor, target_class):
    """Generate Grad-CAM heatmap"""
    model.eval()
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients and activations
    gradients = model.get_activations_gradient()
    activations = model.get_activations(input_tensor).detach()
    
    # Pool gradients across spatial dimensions
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight activations by gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # Create heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap
```

## 🎯 Performance Optimization

### 1. **Model Compression**
```python
# Pruning
import torch.nn.utils.prune as prune

# Structured pruning
prune.ln_structured(model.conv1, name="weight", amount=0.2, n=2, dim=0)

# Unstructured pruning
prune.l1_unstructured(model.linear, name="weight", amount=0.3)

# Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 2. **Knowledge Distillation**
```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """Knowledge distillation loss"""
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    
    # Distillation loss
    distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Student loss
    student_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * distill_loss + (1 - alpha) * student_loss
```

## 📈 Evaluation Metrics

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_classification(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
```

## 🛠️ Practical Implementation Tips

### 1. **Model Architecture Design**
- Start simple, then add complexity
- Use skip connections for deep networks
- Apply batch normalization after convolutions
- Use dropout for regularization
- Consider attention mechanisms for long sequences

### 2. **Training Best Practices**
- Initialize weights properly (Xavier/He initialization)
- Use learning rate warmup for large models
- Monitor gradient norms to detect vanishing/exploding gradients
- Save checkpoints regularly
- Use early stopping to prevent overfitting

### 3. **Debugging Deep Networks**
```python
# Monitor gradient norms
def monitor_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# Check for dead neurons
def check_dead_neurons(model, dataloader):
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(hook_fn(name))
    
    # Forward pass
    with torch.no_grad():
        for inputs, _ in dataloader:
            _ = model(inputs)
            break
    
    # Check for dead neurons
    for name, activation in activations.items():
        dead_ratio = (activation == 0).float().mean()
        print(f"{name}: {dead_ratio:.2%} dead neurons")
```

## 🎓 Learning Path

### Week 1: Foundations
- Multi-layer perceptrons
- Backpropagation algorithm
- Basic regularization

### Week 2: Convolutional Networks
- CNN fundamentals
- Classic architectures
- Transfer learning

### Week 3: Sequential Models
- RNNs and LSTM
- Sequence-to-sequence models
- Attention mechanisms

### Week 4: Advanced Topics
- Modern architectures
- Optimization techniques
- Model deployment

## 📚 Additional Resources

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with PyTorch" by Eli Stevens

### Papers
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Attention Is All You Need" (Transformer)
- "Batch Normalization: Accelerating Deep Network Training"

### Online Courses
- CS231n: Convolutional Neural Networks (Stanford)
- CS224n: Natural Language Processing (Stanford)
- Deep Learning Specialization (Coursera)

---

**Next Module**: [Computer Vision with CNNs](../03-Computer-Vision/) →

*Ready to dive deep into neural networks? Let's build some amazing architectures! 🚀*