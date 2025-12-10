# 👁️ Computer Vision with CNNs

## 🎯 Learning Objectives
- Understand computer vision fundamentals and image processing
- Master Convolutional Neural Networks (CNNs) architecture
- Implement image classification, object detection, and segmentation
- Learn transfer learning and fine-tuning techniques
- Build practical computer vision applications

## 📚 Table of Contents
1. [Computer Vision Fundamentals](#fundamentals)
2. [Image Processing Basics](#image-processing)
3. [Convolutional Neural Networks](#cnns)
4. [CNN Architectures](#architectures)
5. [Transfer Learning](#transfer-learning)
6. [Object Detection](#object-detection)
7. [Image Segmentation](#segmentation)
8. [Practical Applications](#applications)

## 🔬 Computer Vision Fundamentals {#fundamentals}

### Digital Images and Representation
```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
import seaborn as sns

# Understanding digital images
def explore_image_properties():
    """Explore basic properties of digital images"""
    
    # Load a sample image
    digits = load_digits()
    sample_image = digits.images[0]
    
    print("Image Properties:")
    print(f"Shape: {sample_image.shape}")
    print(f"Data type: {sample_image.dtype}")
    print(f"Min value: {sample_image.min()}")
    print(f"Max value: {sample_image.max()}")
    print(f"Mean value: {sample_image.mean():.2f}")
    
    # Visualize image and its pixel values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(sample_image, cmap='gray')
    axes[0].set_title('Original Image (8x8)')
    axes[0].axis('off')
    
    # Pixel values as heatmap
    sns.heatmap(sample_image, annot=True, fmt='.0f', cmap='gray', ax=axes[1])
    axes[1].set_title('Pixel Values')
    
    # 3D surface plot
    x, y = np.meshgrid(range(8), range(8))
    axes[2] = plt.subplot(1, 3, 3, projection='3d')
    axes[2].plot_surface(x, y, sample_image, cmap='viridis')
    axes[2].set_title('3D Representation')
    
    plt.tight_layout()
    plt.show()

explore_image_properties()

# Color images (RGB)
def demonstrate_color_channels():
    """Demonstrate RGB color channels"""
    
    # Create a simple color image
    height, width = 100, 100
    
    # Red channel
    red_channel = np.zeros((height, width, 3))
    red_channel[:, :width//3, 0] = 255
    
    # Green channel  
    green_channel = np.zeros((height, width, 3))
    green_channel[:, width//3:2*width//3, 1] = 255
    
    # Blue channel
    blue_channel = np.zeros((height, width, 3))
    blue_channel[:, 2*width//3:, 2] = 255
    
    # Combined image
    combined = red_channel + green_channel + blue_channel
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(red_channel.astype(np.uint8))
    axes[0, 0].set_title('Red Channel')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(green_channel.astype(np.uint8))
    axes[0, 1].set_title('Green Channel')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(blue_channel.astype(np.uint8))
    axes[1, 0].set_title('Blue Channel')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(combined.astype(np.uint8))
    axes[1, 1].set_title('Combined RGB')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

demonstrate_color_channels()
```

## 🖼️ Image Processing Basics {#image-processing}

### Filtering and Convolution
```python
def demonstrate_convolution():
    """Demonstrate convolution operation with different kernels"""
    
    # Create sample image
    image = np.zeros((20, 20))
    image[5:15, 5:15] = 1  # White square
    image[8:12, 8:12] = 0  # Black square inside
    
    # Define different kernels
    kernels = {
        'Identity': np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]),
        
        'Edge Detection': np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]),
        
        'Sharpen': np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]]),
        
        'Blur': np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]]) / 9,
        
        'Sobel X': np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]),
        
        'Sobel Y': np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
    }
    
    # Apply convolution
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Apply each kernel
    for i, (name, kernel) in enumerate(kernels.items(), 1):
        # Manual convolution
        result = cv2.filter2D(image, -1, kernel)
        
        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'{name}')
        axes[i].axis('off')
    
    # Hide last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()

demonstrate_convolution()

# Advanced image processing
def advanced_image_processing():
    """Demonstrate advanced image processing techniques"""
    
    # Load CIFAR-10 for real images
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    sample_image = x_train[0]
    
    # Convert to grayscale for some operations
    gray_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(sample_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray_image)
    axes[0, 2].imshow(equalized, cmap='gray')
    axes[0, 2].set_title('Histogram Equalized')
    axes[0, 2].axis('off')
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(sample_image, (15, 15), 0)
    axes[0, 3].imshow(blurred)
    axes[0, 3].set_title('Gaussian Blur')
    axes[0, 3].axis('off')
    
    # Edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Canny Edge Detection')
    axes[1, 0].axis('off')
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    axes[1, 1].imshow(opening, cmap='gray')
    axes[1, 1].set_title('Morphological Opening')
    axes[1, 1].axis('off')
    
    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = sample_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    axes[1, 2].imshow(contour_image)
    axes[1, 2].set_title('Contours')
    axes[1, 2].axis('off')
    
    # Histogram
    axes[1, 3].hist(gray_image.flatten(), bins=50, alpha=0.7)
    axes[1, 3].set_title('Pixel Intensity Histogram')
    axes[1, 3].set_xlabel('Pixel Intensity')
    axes[1, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

advanced_image_processing()
```

## 🧠 Convolutional Neural Networks {#cnns}

### CNN Architecture Components
```python
class ConvolutionalLayer:
    """Custom implementation of convolutional layer"""
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters with Xavier initialization
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2.0 / (filter_size * filter_size))
        self.biases = np.zeros(num_filters)
    
    def add_padding(self, image, padding):
        """Add zero padding to image"""
        if padding == 0:
            return image
        return np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    def convolution_operation(self, image, filter_weights):
        """Perform convolution operation"""
        padded_image = self.add_padding(image, self.padding)
        
        # Calculate output dimensions
        output_height = (padded_image.shape[0] - self.filter_size) // self.stride + 1
        output_width = (padded_image.shape[1] - self.filter_size) // self.stride + 1
        
        output = np.zeros((output_height, output_width))
        
        for i in range(0, output_height * self.stride, self.stride):
            for j in range(0, output_width * self.stride, self.stride):
                # Extract region
                region = padded_image[i:i+self.filter_size, j:j+self.filter_size]
                # Convolution
                output[i//self.stride, j//self.stride] = np.sum(region * filter_weights)
        
        return output
    
    def forward(self, input_image):
        """Forward pass through convolutional layer"""
        if len(input_image.shape) == 2:
            # Single channel
            output = np.zeros((self.num_filters, 
                             (input_image.shape[0] + 2*self.padding - self.filter_size) // self.stride + 1,
                             (input_image.shape[1] + 2*self.padding - self.filter_size) // self.stride + 1))
            
            for f in range(self.num_filters):
                output[f] = self.convolution_operation(input_image, self.filters[f]) + self.biases[f]
        
        return output

# Demonstrate CNN components
def demonstrate_cnn_components():
    """Demonstrate different CNN components"""
    
    # Load sample image
    (x_train, _), _ = keras.datasets.mnist.load_data()
    sample_image = x_train[0] / 255.0
    
    # Create convolutional layer
    conv_layer = ConvolutionalLayer(num_filters=4, filter_size=3, stride=1, padding=1)
    
    # Apply convolution
    feature_maps = conv_layer.forward(sample_image)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(sample_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Feature maps
    for i in range(4):
        row = (i + 1) // 3
        col = (i + 1) % 3
        if row < 2 and col < 3:
            axes[row, col].imshow(feature_maps[i], cmap='viridis')
            axes[row, col].set_title(f'Feature Map {i+1}')
            axes[row, col].axis('off')
    
    # Hide empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate pooling
    def max_pooling(feature_map, pool_size=2, stride=2):
        """Max pooling operation"""
        output_height = (feature_map.shape[0] - pool_size) // stride + 1
        output_width = (feature_map.shape[1] - pool_size) // stride + 1
        
        output = np.zeros((output_height, output_width))
        
        for i in range(output_height):
            for j in range(output_width):
                start_i = i * stride
                start_j = j * stride
                region = feature_map[start_i:start_i+pool_size, start_j:start_j+pool_size]
                output[i, j] = np.max(region)
        
        return output
    
    # Apply max pooling to first feature map
    pooled = max_pooling(feature_maps[0])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(feature_maps[0], cmap='viridis')
    axes[0].set_title(f'Before Pooling {feature_maps[0].shape}')
    axes[0].axis('off')
    
    axes[1].imshow(pooled, cmap='viridis')
    axes[1].set_title(f'After Max Pooling {pooled.shape}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

demonstrate_cnn_components()
```

### Complete CNN Implementation
```python
def create_cnn_model(input_shape, num_classes):
    """Create a complete CNN model"""
    
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Third convolutional block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        
        # Classifier
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Train CNN on CIFAR-10
def train_cnn_cifar10():
    """Train CNN on CIFAR-10 dataset"""
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Create model
    model = create_cnn_model((32, 32, 3), 10)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Visualize CNN training
def plot_training_history(history):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage (commented out for notebook compatibility)
# model, history = train_cnn_cifar10()
# plot_training_history(history)
```

## 🏗️ CNN Architectures {#architectures}

### Famous CNN Architectures
```python
def create_lenet5():
    """LeNet-5 architecture (1998)"""
    model = keras.Sequential([
        keras.layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Conv2D(16, (5, 5), activation='tanh'),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='tanh'),
        keras.layers.Dense(84, activation='tanh'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def create_alexnet():
    """AlexNet architecture (2012)"""
    model = keras.Sequential([
        keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1000, activation='softmax')
    ])
    return model

def create_vgg16():
    """VGG-16 architecture (2014)"""
    model = keras.Sequential([
        # Block 1
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 2
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 3
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 4
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 5
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        
        # Classifier
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1000, activation='softmax')
    ])
    return model

# ResNet building blocks
def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    """Residual block for ResNet"""
    
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride)(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = x
    
    x = keras.layers.Conv2D(filters, 1, strides=stride)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(4 * filters, 1)(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Add()([shortcut, x])
    x = keras.layers.Activation('relu')(x)
    
    return x

def create_resnet50():
    """ResNet-50 architecture (2015)"""
    
    inputs = keras.layers.Input(shape=(224, 224, 3))
    
    # Initial conv layer
    x = keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64, conv_shortcut=False)
    x = residual_block(x, 64, conv_shortcut=False)
    
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128, conv_shortcut=False)
    x = residual_block(x, 128, conv_shortcut=False)
    x = residual_block(x, 128, conv_shortcut=False)
    
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256, conv_shortcut=False)
    x = residual_block(x, 256, conv_shortcut=False)
    x = residual_block(x, 256, conv_shortcut=False)
    x = residual_block(x, 256, conv_shortcut=False)
    x = residual_block(x, 256, conv_shortcut=False)
    
    x = residual_block(x, 512, stride=2, conv_shortcut=True)
    x = residual_block(x, 512, conv_shortcut=False)
    x = residual_block(x, 512, conv_shortcut=False)
    
    # Final layers
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Compare architectures
def compare_architectures():
    """Compare different CNN architectures"""
    
    architectures = {
        'LeNet-5': create_lenet5,
        'AlexNet': create_alexnet,
        'VGG-16': create_vgg16,
        'ResNet-50': create_resnet50
    }
    
    comparison_data = []
    
    for name, create_func in architectures.items():
        try:
            model = create_func()
            params = model.count_params()
            layers = len(model.layers)
            
            comparison_data.append({
                'Architecture': name,
                'Parameters': f"{params:,}",
                'Layers': layers
            })
            
            print(f"{name}:")
            print(f"  Parameters: {params:,}")
            print(f"  Layers: {layers}")
            print()
            
        except Exception as e:
            print(f"Error creating {name}: {e}")
    
    return comparison_data

comparison_data = compare_architectures()
```

## 🔄 Transfer Learning {#transfer-learning}

### Pre-trained Models and Fine-tuning
```python
def demonstrate_transfer_learning():
    """Demonstrate transfer learning with pre-trained models"""
    
    # Load pre-trained VGG16 model
    base_model = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    
    print("Transfer Learning Model:")
    model.summary()
    
    return model

def fine_tuning_example():
    """Example of fine-tuning a pre-trained model"""
    
    # Create base model
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add custom head
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune top layers
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    print("Fine-tuning Model:")
    print(f"Total layers: {len(model.layers)}")
    print(f"Trainable layers: {sum(1 for layer in model.layers if layer.trainable)}")
    
    return model

# Feature extraction example
def extract_features_with_pretrained():
    """Extract features using pre-trained CNN"""
    
    # Load pre-trained model without top layers
    feature_extractor = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    
    # Load sample images
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    
    # Resize images to 224x224 (VGG16 input size)
    x_sample = x_train[:100]  # Use small sample
    x_resized = tf.image.resize(x_sample, [224, 224])
    x_resized = keras.applications.vgg16.preprocess_input(x_resized)
    
    # Extract features
    features = feature_extractor.predict(x_resized)
    
    print(f"Original image shape: {x_sample.shape}")
    print(f"Extracted features shape: {features.shape}")
    
    # Visualize feature distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(x_sample[0])
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.hist(features[0], bins=50, alpha=0.7)
    plt.title('Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.plot(features[0])
    plt.title('Feature Vector')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    
    plt.tight_layout()
    plt.show()
    
    return features

# Example usage
transfer_model = demonstrate_transfer_learning()
fine_tune_model = fine_tuning_example()
features = extract_features_with_pretrained()
```

## 🎯 Object Detection {#object-detection}

### Object Detection Fundamentals
```python
def demonstrate_object_detection_concepts():
    """Demonstrate object detection concepts"""
    
    # Simulate bounding box operations
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def non_max_suppression(boxes, scores, threshold=0.5):
        """Non-Maximum Suppression"""
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = [calculate_iou(boxes[current], boxes[i]) for i in indices[1:]]
            
            # Keep boxes with IoU below threshold
            indices = [indices[i+1] for i, iou in enumerate(ious) if iou < threshold]
        
        return keep
    
    # Example boxes and scores
    boxes = np.array([
        [10, 10, 50, 50],   # Box 1
        [15, 15, 55, 55],   # Box 2 (overlaps with Box 1)
        [100, 100, 140, 140], # Box 3
        [105, 105, 145, 145]  # Box 4 (overlaps with Box 3)
    ])
    
    scores = np.array([0.9, 0.8, 0.85, 0.7])
    
    # Apply NMS
    keep_indices = non_max_suppression(boxes, scores, threshold=0.3)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before NMS
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.invert_yaxis()
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (box, score) in enumerate(zip(boxes, scores)):
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                           fill=False, color=colors[i], linewidth=2)
        ax1.add_patch(rect)
        ax1.text(box[0], box[1]-5, f'Score: {score:.2f}', color=colors[i])
    
    ax1.set_title('Before NMS')
    ax1.grid(True, alpha=0.3)
    
    # After NMS
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 200)
    ax2.invert_yaxis()
    
    for i in keep_indices:
        box = boxes[i]
        score = scores[i]
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                           fill=False, color=colors[i], linewidth=2)
        ax2.add_patch(rect)
        ax2.text(box[0], box[1]-5, f'Score: {score:.2f}', color=colors[i])
    
    ax2.set_title('After NMS')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original boxes: {len(boxes)}")
    print(f"After NMS: {len(keep_indices)}")
    print(f"Kept indices: {keep_indices}")

demonstrate_object_detection_concepts()

# Simple object detection with sliding window
def sliding_window_detection():
    """Demonstrate sliding window object detection"""
    
    # Create synthetic image with objects
    image = np.zeros((200, 200, 3))
    
    # Add some "objects" (colored rectangles)
    image[50:80, 50:80] = [1, 0, 0]  # Red square
    image[120:150, 120:150] = [0, 1, 0]  # Green square
    image[30:60, 140:170] = [0, 0, 1]  # Blue square
    
    # Define sliding window
    window_size = 40
    stride = 20
    
    detections = []
    
    # Slide window across image
    for y in range(0, image.shape[0] - window_size, stride):
        for x in range(0, image.shape[1] - window_size, stride):
            window = image[y:y+window_size, x:x+window_size]
            
            # Simple detection: check if window contains significant color
            if np.max(window) > 0.5:
                confidence = np.max(window)
                detections.append((x, y, x+window_size, y+window_size, confidence))
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Detections
    ax2.imshow(image)
    for detection in detections:
        x1, y1, x2, y2, conf = detection
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='yellow', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f'{conf:.2f}', color='yellow', fontsize=8)
    
    ax2.set_title(f'Sliding Window Detections ({len(detections)} windows)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

sliding_window_detection()
```

## 🎨 Image Segmentation {#segmentation}

### Semantic Segmentation
```python
def create_unet_model(input_size=(256, 256, 3), num_classes=1):
    """Create U-Net model for image segmentation"""
    
    inputs = keras.Input(input_size)
    
    # Encoder (Contracting Path)
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (Expansive Path)
    u6 = keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output
    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Demonstrate segmentation
def demonstrate_segmentation():
    """Demonstrate image segmentation concepts"""
    
    # Create synthetic segmentation data
    def create_synthetic_data(num_samples=100, img_size=128):
        images = []
        masks = []
        
        for _ in range(num_samples):
            # Create image with geometric shapes
            img = np.zeros((img_size, img_size, 3))
            mask = np.zeros((img_size, img_size, 1))
            
            # Add random circles and rectangles
            num_shapes = np.random.randint(1, 4)
            
            for _ in range(num_shapes):
                if np.random.random() > 0.5:
                    # Circle
                    center = (np.random.randint(20, img_size-20), 
                             np.random.randint(20, img_size-20))
                    radius = np.random.randint(10, 30)
                    color = np.random.random(3)
                    
                    y, x = np.ogrid[:img_size, :img_size]
                    circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                    
                    img[circle_mask] = color
                    mask[circle_mask] = 1
                else:
                    # Rectangle
                    x1, y1 = np.random.randint(0, img_size//2, 2)
                    x2, y2 = np.random.randint(img_size//2, img_size, 2)
                    color = np.random.random(3)
                    
                    img[y1:y2, x1:x2] = color
                    mask[y1:y2, x1:x2] = 1
            
            images.append(img)
            masks.append(mask)
        
        return np.array(images), np.array(masks)
    
    # Create data
    X, y = create_synthetic_data(num_samples=20, img_size=64)
    
    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Original image
        axes[0, i].imshow(X[i])
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Mask
        axes[1, i].imshow(y[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return X, y

# Create and visualize U-Net
unet_model = create_unet_model(input_size=(64, 64, 3))
print("U-Net Architecture:")
print(f"Total parameters: {unet_model.count_params():,}")

# Visualize architecture
keras.utils.plot_model(unet_model, show_shapes=True, show_layer_names=True, dpi=150)

# Generate synthetic data
X_seg, y_seg = demonstrate_segmentation()
```

## 🚀 Practical Applications {#applications}

### Complete Computer Vision Pipeline
```python
class ComputerVisionPipeline:
    """Complete computer vision pipeline"""
    
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.postprocessor = None
    
    def preprocess_image(self, image):
        """Preprocess input image"""
        # Resize
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
    def load_model(self, model_path=None):
        """Load pre-trained model"""
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            # Use pre-trained ResNet50
            self.model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=True
            )
    
    def predict(self, image):
        """Make prediction on image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed_image)
        
        # Postprocess
        if self.model.name == 'resnet50':
            # Decode ImageNet predictions
            decoded = keras.applications.resnet50.decode_predictions(predictions, top=5)[0]
            return decoded
        else:
            return predictions
    
    def visualize_prediction(self, image, predictions):
        """Visualize predictions"""
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Predictions
        plt.subplot(1, 2, 2)
        if isinstance(predictions[0], tuple):  # ImageNet format
            labels = [pred[1] for pred in predictions]
            scores = [pred[2] for pred in predictions]
            
            y_pos = np.arange(len(labels))
            plt.barh(y_pos, scores)
            plt.yticks(y_pos, labels)
            plt.xlabel('Confidence')
            plt.title('Top 5 Predictions')
        else:
            plt.plot(predictions[0])
            plt.title('Prediction Scores')
            plt.xlabel('Class')
            plt.ylabel('Probability')
        
        plt.tight_layout()
        plt.show()

# Example usage
def demonstrate_cv_pipeline():
    """Demonstrate complete CV pipeline"""
    
    # Initialize pipeline
    pipeline = ComputerVisionPipeline()
    pipeline.load_model()
    
    # Load sample image from CIFAR-10
    (x_test, y_test), _ = keras.datasets.cifar10.load_data()
    sample_image = x_test[0]
    
    # Make prediction
    predictions = pipeline.predict(sample_image)
    
    # Visualize
    pipeline.visualize_prediction(sample_image, predictions)
    
    print("Top 5 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i+1}. {label}: {score:.4f}")

# Run demonstration
demonstrate_cv_pipeline()

# Performance optimization tips
def optimization_tips():
    """Demonstrate performance optimization techniques"""
    
    print("Computer Vision Optimization Tips:")
    print("=" * 50)
    
    tips = [
        "1. Use appropriate input resolution (don't always use 224x224)",
        "2. Apply data augmentation during training",
        "3. Use transfer learning when possible",
        "4. Implement proper preprocessing pipelines",
        "5. Use mixed precision training for faster training",
        "6. Batch predictions for better throughput",
        "7. Use TensorRT or TensorFlow Lite for deployment",
        "8. Implement proper caching for repeated operations",
        "9. Use GPU acceleration when available",
        "10. Profile your code to identify bottlenecks"
    ]
    
    for tip in tips:
        print(tip)
    
    # Demonstrate batch processing
    print("\nBatch Processing Example:")
    
    # Load multiple images
    (x_test, _), _ = keras.datasets.cifar10.load_data()
    batch_images = x_test[:32]  # Batch of 32 images
    
    # Resize for ResNet50
    batch_resized = np.array([cv2.resize(img, (224, 224)) for img in batch_images])
    batch_resized = batch_resized.astype('float32') / 255.0
    
    # Time single vs batch prediction
    import time
    
    model = keras.applications.ResNet50(weights='imagenet')
    
    # Single predictions
    start_time = time.time()
    for img in batch_resized:
        _ = model.predict(np.expand_dims(img, axis=0), verbose=0)
    single_time = time.time() - start_time
    
    # Batch prediction
    start_time = time.time()
    _ = model.predict(batch_resized, verbose=0)
    batch_time = time.time() - start_time
    
    print(f"Single predictions time: {single_time:.2f} seconds")
    print(f"Batch prediction time: {batch_time:.2f} seconds")
    print(f"Speedup: {single_time/batch_time:.2f}x")

optimization_tips()
```

## 🎯 Key Takeaways

1. **Computer Vision** transforms visual data into actionable insights
2. **CNNs** are specifically designed for spatial data processing
3. **Transfer Learning** accelerates development with pre-trained models
4. **Data Augmentation** improves model generalization
5. **Object Detection** extends beyond classification to localization
6. **Segmentation** provides pixel-level understanding
7. **Optimization** is crucial for real-world deployment

## 📚 Next Steps

- Explore [Advanced Generative Models](../../03-Generative-AI/)
- Learn about [Natural Language Processing](../04-NLP/)
- Practice with [Computer Vision Projects](../../Projects/)
- Study [Multimodal AI](../../04-Advanced/01-Multimodal-AI/)

## 🔗 Additional Resources

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning for Computer Vision with Python](https://www.pyimagesearch.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Computer Vision Tutorials](https://www.tensorflow.org/tutorials/images)
- [Papers With Code - Computer Vision](https://paperswithcode.com/area/computer-vision)