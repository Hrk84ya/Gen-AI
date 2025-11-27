# 🖼️ Project 1: Image Classifier

## 🎯 Project Overview
Build a deep learning model to classify images with high accuracy using Convolutional Neural Networks (CNNs).

**Difficulty**: 🌱 Beginner  
**Time**: 1-2 weeks  
**Framework**: TensorFlow/Keras

## 🎓 What You'll Learn
- Data preprocessing and augmentation
- CNN architecture design
- Model training and evaluation
- Transfer learning
- Model deployment

## 📋 Project Requirements

### Technical Requirements
- Python 3.8+
- TensorFlow 2.x
- 4GB+ RAM
- GPU recommended (but not required)

### Dataset Options
1. **CIFAR-10** (Beginner): 10 classes, 32x32 images
2. **Fashion-MNIST** (Beginner): 10 clothing categories
3. **Custom Dataset** (Intermediate): Your own images

## 🚀 Getting Started

### Step 1: Setup Environment
```bash
# Create virtual environment
python -m venv image-classifier-env
source image-classifier-env/bin/activate

# Install dependencies
pip install tensorflow matplotlib numpy pandas scikit-learn
```

### Step 2: Load and Explore Data
```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Explore data
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Visualize samples
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()
```

### Step 3: Data Preprocessing
```python
# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(x_train)
```

### Step 4: Build CNN Model
```python
def create_cnn_model():
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
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
        keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

# Create and compile model
model = create_cnn_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Step 5: Train the Model
```python
# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)
```

### Step 6: Evaluate and Visualize Results
```python
# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🚀 Advanced Features

### Transfer Learning with Pre-trained Models
```python
# Use pre-trained ResNet50
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Model Interpretation
```python
import tensorflow as tf

# Grad-CAM for visualization
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

## 📊 Project Deliverables

### 1. Jupyter Notebook
- Data exploration and visualization
- Model architecture explanation
- Training process documentation
- Results analysis and interpretation

### 2. Python Scripts
- `data_loader.py`: Data loading and preprocessing
- `model.py`: Model architecture definitions
- `train.py`: Training script
- `evaluate.py`: Evaluation and testing

### 3. Web Application (Optional)
```python
import streamlit as st
from PIL import Image
import numpy as np

st.title('Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    img_array = np.array(image.resize((32, 32))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f'Prediction: {predicted_class}')
    st.write(f'Confidence: {confidence:.2f}')
```

## 🎯 Success Metrics

### Minimum Requirements
- [ ] Model achieves >70% accuracy on test set
- [ ] Proper data preprocessing implemented
- [ ] Training/validation curves plotted
- [ ] Code is well-documented

### Stretch Goals
- [ ] Achieve >85% accuracy
- [ ] Implement transfer learning
- [ ] Create web application
- [ ] Add model interpretation features
- [ ] Deploy model to cloud

## 🔍 Evaluation Rubric

| Criteria | Excellent (4) | Good (3) | Fair (2) | Needs Work (1) |
|----------|---------------|----------|----------|----------------|
| **Code Quality** | Clean, documented, modular | Well-structured | Some organization | Messy, undocumented |
| **Model Performance** | >85% accuracy | 75-85% accuracy | 65-75% accuracy | <65% accuracy |
| **Data Preprocessing** | Comprehensive augmentation | Basic augmentation | Normalization only | Minimal preprocessing |
| **Analysis** | Deep insights, visualizations | Good analysis | Basic analysis | Limited analysis |

## 🚀 Next Steps

After completing this project:
1. Try different architectures (ResNet, DenseNet, EfficientNet)
2. Experiment with different datasets
3. Learn about object detection
4. Move to [Project 2: Chatbot](../02-Chatbot/)

## 📚 Resources

### Tutorials
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Keras Documentation](https://keras.io/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

### Datasets
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](http://www.image-net.org/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

**Ready to start?** Clone the starter code and begin your first AI project! 🚀

*Remember: The goal is learning, not perfection. Start simple and iterate!*