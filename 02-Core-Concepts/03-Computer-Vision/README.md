# Computer Vision with Deep Learning

## 🎯 Learning Objectives
By the end of this module, you will:
- Master convolutional neural networks (CNNs) for image analysis
- Understand modern computer vision architectures
- Implement object detection and segmentation models
- Apply transfer learning for computer vision tasks
- Build end-to-end computer vision applications
- Understand vision transformers and modern approaches

## 👁️ What is Computer Vision?

Computer vision enables machines to interpret and understand visual information from the world. Deep learning has revolutionized this field, achieving human-level performance on many visual tasks.

### Key Applications:
- **Image Classification**: Categorizing images into classes
- **Object Detection**: Locating and identifying objects in images
- **Semantic Segmentation**: Pixel-level classification
- **Instance Segmentation**: Separating individual object instances
- **Face Recognition**: Identifying individuals from facial features
- **Medical Imaging**: Analyzing X-rays, MRIs, CT scans
- **Autonomous Vehicles**: Understanding road scenes
- **Augmented Reality**: Overlaying digital content on real world

## 📚 Module Contents

### 1. [CNN Fundamentals](./01_CNN_fundamentals.ipynb)
- Convolution and pooling operations
- Feature maps and receptive fields
- CNN building blocks
- Classic architectures (LeNet, AlexNet, VGG)

### 2. [Modern CNN Architectures](./02_modern_architectures.ipynb)
- ResNet and skip connections
- DenseNet and feature reuse
- EfficientNet and compound scaling
- MobileNet for mobile deployment

### 3. [Object Detection](./03_object_detection.ipynb)
- R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
- YOLO (You Only Look Once)
- SSD (Single Shot Detector)
- Modern detectors (DETR, EfficientDet)

### 4. [Image Segmentation](./04_image_segmentation.ipynb)
- Semantic segmentation with FCN
- U-Net for biomedical images
- DeepLab and atrous convolutions
- Instance segmentation with Mask R-CNN

### 5. [Transfer Learning](./05_transfer_learning.ipynb)
- Pre-trained model usage
- Feature extraction vs fine-tuning
- Domain adaptation techniques
- Few-shot learning approaches

### 6. [Vision Transformers](./06_vision_transformers.ipynb)
- Vision Transformer (ViT) architecture
- Patch embeddings and positional encoding
- Hybrid CNN-Transformer models
- DETR for object detection

### 7. [Advanced Topics](./07_advanced_topics.ipynb)
- Generative models for images (GANs, VAEs)
- Style transfer and image synthesis
- 3D computer vision
- Video analysis and action recognition

## 🏗️ CNN Architecture Evolution

### Historical Timeline
```
1989: LeNet-5 (Handwritten digit recognition)
2012: AlexNet (ImageNet breakthrough)
2014: VGG (Very deep networks)
2014: GoogLeNet/Inception (Efficient architectures)
2015: ResNet (Residual connections)
2017: DenseNet (Dense connections)
2019: EfficientNet (Compound scaling)
2020: Vision Transformer (Attention for vision)
2021: Swin Transformer (Hierarchical vision transformer)
```

## 🔧 Core Concepts

### 1. **Convolution Operation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolution
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

# Different types of convolutions
convolutions = {
    'standard': nn.Conv2d(64, 128, 3, padding=1),
    'depthwise': nn.Conv2d(64, 64, 3, padding=1, groups=64),
    'pointwise': nn.Conv2d(64, 128, 1),
    'dilated': nn.Conv2d(64, 128, 3, padding=2, dilation=2),
    'transposed': nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
}
```

### 2. **Pooling Operations**
```python
# Different pooling methods
pooling_layers = {
    'max_pool': nn.MaxPool2d(2, stride=2),
    'avg_pool': nn.AvgPool2d(2, stride=2),
    'adaptive_avg': nn.AdaptiveAvgPool2d((1, 1)),
    'adaptive_max': nn.AdaptiveMaxPool2d((1, 1))
}

# Global average pooling (common in modern architectures)
def global_avg_pool2d(x):
    return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
```

### 3. **Modern CNN Block Designs**
```python
class ResidualBlock(nn.Module):
    """ResNet-style residual block"""
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
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class DenseBlock(nn.Module):
    """DenseNet-style dense block"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

## 🎯 Popular Architectures

### 1. **ResNet Implementation**
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Create ResNet-18
def resnet18(num_classes=1000):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)
```

### 2. **EfficientNet Building Blocks**
```python
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            self.se = SEBlock(expanded_channels, int(in_channels * se_ratio))
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        residual = x
        
        # Expansion
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        # Depthwise + SE
        x = self.depthwise_conv(x)
        if hasattr(self, 'se'):
            x = self.se(x)
        
        # Projection
        x = self.project_conv(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
        
        return x
```

### 3. **Vision Transformer (ViT)**
```python
class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)
        # Flatten patches: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True
            ),
            num_layers=depth
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Use class token
        x = self.head(x)
        
        return x
```

## 🎯 Object Detection

### 1. **YOLO Architecture**
```python
class YOLOv5(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone (simplified)
        self.backbone = nn.Sequential(
            # Focus layer
            nn.Conv2d(12, 64, 3, padding=1),  # 4*3 channels from focus
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Additional layers would go here...
        )
        
        # Neck (FPN-like structure)
        self.neck = nn.Sequential(
            # Feature pyramid network layers
        )
        
        # Head (detection layers)
        self.head = nn.ModuleList([
            nn.Conv2d(256, (num_classes + 5) * 3, 1)  # 3 anchors per grid
            for _ in range(3)  # 3 detection scales
        ])
    
    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Neck processing
        fpn_features = self.neck(features)
        
        # Detection head
        detections = []
        for i, head in enumerate(self.head):
            detection = head(fpn_features[i])
            detections.append(detection)
        
        return detections

def yolo_loss(predictions, targets, anchors):
    """YOLO loss function"""
    # Objectness loss
    obj_loss = F.binary_cross_entropy_with_logits(
        predictions[..., 4], targets[..., 4]
    )
    
    # Classification loss
    cls_loss = F.cross_entropy(
        predictions[..., 5:], targets[..., 5:].long()
    )
    
    # Bounding box regression loss
    box_loss = F.mse_loss(
        predictions[..., :4], targets[..., :4]
    )
    
    return obj_loss + cls_loss + box_loss
```

### 2. **Faster R-CNN Components**
```python
class RPN(nn.Module):
    """Region Proposal Network"""
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, 1)
    
    def forward(self, features):
        x = F.relu(self.conv(features))
        
        objectness = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)
        
        return objectness, bbox_deltas

class ROIHead(nn.Module):
    """ROI Head for final classification and regression"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
    
    def forward(self, roi_features):
        x = roi_features.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        cls_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        return cls_scores, bbox_deltas
```

## 🖼️ Image Segmentation

### 1. **U-Net Architecture**
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.conv_block(d4.size(1), 512)(d4)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.conv_block(d3.size(1), 256)(d3)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.conv_block(d2.size(1), 128)(d2)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.conv_block(d1.size(1), 64)(d1)
        
        return self.final(d1)
```

## 📊 Evaluation Metrics

### 1. **Classification Metrics**
```python
def calculate_accuracy(outputs, labels, topk=(1, 5)):
    """Calculate top-k accuracy"""
    maxk = max(topk)
    batch_size = labels.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def confusion_matrix_metrics(cm):
    """Calculate metrics from confusion matrix"""
    # Per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Overall metrics
    accuracy = np.trace(cm) / np.sum(cm)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
```

### 2. **Detection Metrics**
```python
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU)"""
    # box format: [x1, y1, x2, y2]
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

def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """Calculate mean Average Precision (mAP)"""
    # Implementation would involve:
    # 1. For each class, calculate precision-recall curve
    # 2. Calculate Average Precision (AP) for each class
    # 3. Return mean of all APs
    pass
```

### 3. **Segmentation Metrics**
```python
def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for segmentation"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU for segmentation"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou
```

## 🛠️ Practical Tips

### 1. **Data Preprocessing**
```python
# Standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 2. **Transfer Learning Best Practices**
```python
def setup_transfer_learning(model, num_classes, freeze_backbone=True):
    """Setup model for transfer learning"""
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    return model

# Different learning rates for different parts
def get_optimizer_with_different_lrs(model, backbone_lr=1e-4, head_lr=1e-3):
    """Use different learning rates for backbone and head"""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ])
    
    return optimizer
```

### 3. **Model Deployment Optimization**
```python
# Model quantization for deployment
def quantize_model(model, calibration_loader):
    """Quantize model for faster inference"""
    model.eval()
    
    # Post-training quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    return quantized_model

# TensorRT optimization (if available)
def optimize_with_tensorrt(model, input_shape):
    """Optimize model with TensorRT"""
    try:
        import torch_tensorrt
        
        # Compile with TensorRT
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(input_shape)],
            enabled_precisions={torch.float, torch.half}
        )
        
        return trt_model
    except ImportError:
        print("TensorRT not available")
        return model
```

## 🎓 Learning Path

### Week 1: CNN Fundamentals
- Convolution and pooling operations
- Basic CNN architectures
- Image classification project

### Week 2: Modern Architectures
- ResNet, DenseNet, EfficientNet
- Transfer learning techniques
- Performance optimization

### Week 3: Object Detection
- YOLO and R-CNN families
- Detection metrics and evaluation
- Real-time detection systems

### Week 4: Advanced Topics
- Image segmentation
- Vision transformers
- Deployment and optimization

---

**Next Module**: [Natural Language Processing](../04-NLP/) →

*Ready to see the world through AI eyes? Let's build some amazing computer vision systems! 👁️🤖*