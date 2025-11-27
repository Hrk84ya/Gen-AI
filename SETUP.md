# 🛠️ Setup Guide: Getting Your Environment Ready

## 🎯 Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/Hrk84ya/Gen-AI.git
cd Gen-AI

# 2. Create virtual environment
python -m venv genai-env
source genai-env/bin/activate  # On Windows: genai-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Lab
jupyter lab
```

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional but recommended for advanced projects

### Knowledge Prerequisites
- Basic programming concepts
- High school level mathematics
- Familiarity with command line (helpful but not required)

## 🐍 Python Environment Setup

### Option 1: Anaconda (Recommended for Beginners)
```bash
# Download and install Anaconda from https://www.anaconda.com/

# Create environment
conda create -n genai python=3.9
conda activate genai

# Install packages
conda install jupyter pandas numpy matplotlib scikit-learn
pip install -r requirements.txt
```

### Option 2: Virtual Environment (Lightweight)
```bash
# Create virtual environment
python -m venv genai-env

# Activate environment
# On Windows:
genai-env\Scripts\activate
# On macOS/Linux:
source genai-env/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 3: Poetry (For Advanced Users)
```bash
# Install Poetry: https://python-poetry.org/docs/#installation

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## 🚀 Development Environment Options

### Option 1: Local Setup
**Pros**: Full control, no internet required, use your own hardware
**Cons**: Setup complexity, hardware limitations

```bash
# Install Jupyter Lab
pip install jupyterlab

# Install extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Launch
jupyter lab
```

### Option 2: Google Colab (Recommended for Beginners)
**Pros**: No setup required, free GPU access, easy sharing
**Cons**: Session limits, internet required

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload notebooks or connect to GitHub
3. Enable GPU: Runtime → Change runtime type → GPU

### Option 3: Cloud Platforms
**AWS SageMaker**, **Azure ML**, **Google Cloud AI Platform**
**Pros**: Scalable, professional tools, managed infrastructure
**Cons**: Cost, complexity

## 🔧 Essential Tools Installation

### Code Editors
```bash
# VS Code (Recommended)
# Download from: https://code.visualstudio.com/

# Install Python extension
# Install Jupyter extension
```

### Git Setup
```bash
# Install Git: https://git-scm.com/

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clone repository
git clone https://github.com/Hrk84ya/Gen-AI.git
```

### GPU Setup (Optional but Recommended)

#### NVIDIA GPU Setup
```bash
# Check GPU
nvidia-smi

# Install CUDA Toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

#### Apple Silicon (M1/M2) Setup
```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Verify MPS access
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 📦 Package Installation Guide

### Core ML Libraries
```bash
# Data manipulation
pip install numpy pandas matplotlib seaborn

# Machine learning
pip install scikit-learn

# Deep learning
pip install tensorflow torch torchvision

# Computer vision
pip install opencv-python pillow

# Natural language processing
pip install nltk spacy transformers

# Jupyter and visualization
pip install jupyter jupyterlab plotly
```

### Generative AI Specific
```bash
# Hugging Face ecosystem
pip install transformers datasets accelerate

# Diffusion models
pip install diffusers

# OpenAI API
pip install openai

# LangChain for AI applications
pip install langchain

# Weights & Biases for experiment tracking
pip install wandb
```

## 🧪 Verification Tests

### Test Your Setup
Create a file `test_setup.py`:

```python
# Test basic imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✅ Basic libraries imported successfully")
except ImportError as e:
    print(f"❌ Error importing basic libraries: {e}")

# Test deep learning frameworks
try:
    import tensorflow as tf
    import torch
    print("✅ Deep learning frameworks imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"❌ Error importing deep learning frameworks: {e}")

# Test GPU availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA GPU available: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        print("✅ Apple MPS available")
    else:
        print("⚠️ No GPU acceleration available (CPU only)")
except:
    print("⚠️ Could not check GPU availability")

# Test Jupyter
try:
    import jupyter
    print("✅ Jupyter installed successfully")
except ImportError:
    print("❌ Jupyter not installed")

print("\n🎉 Setup verification complete!")
```

Run the test:
```bash
python test_setup.py
```

## 🔧 Troubleshooting Common Issues

### Issue: Package Installation Fails
```bash
# Solution 1: Upgrade pip
pip install --upgrade pip

# Solution 2: Use conda instead
conda install package-name

# Solution 3: Install from source
pip install git+https://github.com/package/repo.git
```

### Issue: Jupyter Kernel Not Found
```bash
# Install kernel
python -m ipykernel install --user --name genai --display-name "Gen-AI"

# List kernels
jupyter kernelspec list

# Remove kernel if needed
jupyter kernelspec remove genai
```

### Issue: CUDA/GPU Not Working
```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Memory Errors
```bash
# Increase virtual memory (Windows)
# System Properties → Advanced → Performance Settings → Advanced → Virtual Memory

# Monitor memory usage
pip install psutil
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')"
```

## 🌐 Cloud Setup Options

### Google Colab Pro
- **Cost**: $10/month
- **Benefits**: Priority access, longer runtimes, more memory
- **Setup**: No installation required

### AWS SageMaker
```bash
# Create SageMaker notebook instance
# Choose ml.t3.medium for basic work
# Choose ml.p3.2xlarge for GPU work
```

### Kaggle Notebooks
- **Cost**: Free
- **Benefits**: 30 hours/week GPU, datasets included
- **Setup**: Upload notebooks to Kaggle

## 📱 Mobile Development (Optional)

### Jupyter on iPad
1. Install **Carnets** app from App Store
2. Import notebooks via Files app
3. Limited functionality but good for learning

### GitHub Codespaces
1. Open repository in GitHub
2. Click "Code" → "Codespaces" → "Create codespace"
3. Full VS Code environment in browser

## 🔐 API Keys Setup

### OpenAI API
```bash
# Get API key from https://platform.openai.com/
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Hugging Face
```bash
# Get token from https://huggingface.co/settings/tokens
huggingface-cli login

# Or set environment variable
export HUGGINGFACE_HUB_TOKEN="your-token-here"
```

### Weights & Biases
```bash
# Get API key from https://wandb.ai/authorize
wandb login
```

## 📚 Additional Resources

### Documentation
- [Python.org](https://www.python.org/doc/)
- [Jupyter Documentation](https://jupyter.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Guides](https://www.tensorflow.org/guide)

### Communities
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/machine-learning)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)

## ✅ Setup Checklist

Before starting your learning journey, ensure you have:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All required packages installed
- [ ] Jupyter Lab working
- [ ] GPU access verified (if available)
- [ ] Git configured
- [ ] Test script runs successfully
- [ ] API keys configured (if needed)

## 🆘 Getting Help

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Search existing issues** in the repository
3. **Ask in discussions** with detailed error messages
4. **Join our community** for real-time help

---

**🎉 Congratulations!** Your environment is ready. Now head to the [Learning Roadmap](./ROADMAP.md) to start your journey!

*Remember: A good setup is the foundation of productive learning. Take time to get it right! 🚀*