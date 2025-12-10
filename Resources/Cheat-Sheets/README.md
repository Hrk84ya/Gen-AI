# 📚 Generative AI Cheat Sheets

Quick reference guides for key concepts, formulas, and implementations in Generative AI.

## 🎯 Available Cheat Sheets

### 📊 Fundamentals
- [**Probability & Statistics**](./probability-statistics.md) - Essential probability concepts for generative modeling
- [**Linear Algebra**](./linear-algebra.md) - Matrix operations, eigenvalues, and transformations
- [**Calculus & Optimization**](./calculus-optimization.md) - Derivatives, gradients, and optimization algorithms

### 🧠 Neural Networks
- [**Neural Network Basics**](./neural-networks.md) - Perceptrons, MLPs, activation functions
- [**Backpropagation**](./backpropagation.md) - Step-by-step gradient computation
- [**CNN Architectures**](./cnn-architectures.md) - Convolutional layers, pooling, popular architectures

### 🎨 Generative Models
- [**GANs Quick Reference**](./gans-reference.md) - Generator, discriminator, training tips
- [**VAE Essentials**](./vae-essentials.md) - Encoder, decoder, reparameterization trick
- [**Transformer Architecture**](./transformers.md) - Attention mechanism, positional encoding
- [**Diffusion Models**](./diffusion-models.md) - Forward/reverse process, denoising

### 🛠️ Implementation
- [**TensorFlow/Keras**](./tensorflow-keras.md) - Common patterns and best practices
- [**PyTorch**](./pytorch.md) - Essential operations and model building
- [**Hugging Face**](./hugging-face.md) - Transformers library usage

### 📈 Evaluation & Metrics
- [**Generative Model Metrics**](./evaluation-metrics.md) - FID, IS, BLEU, ROUGE
- [**Training Diagnostics**](./training-diagnostics.md) - Loss curves, convergence issues
- [**Hyperparameter Tuning**](./hyperparameter-tuning.md) - Grid search, random search, Bayesian optimization

## 🎨 Format Examples

### Mathematical Formulas
```markdown
## Gaussian Distribution
**PDF**: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

**Parameters**: 
- μ (mu): mean
- σ (sigma): standard deviation
```

### Code Snippets
```python
# Quick implementation example
import torch
import torch.nn as nn

class SimpleGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
```

### Quick Reference Tables
| Activation | Formula | Range | Use Case |
|------------|---------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Hidden layers |
| Sigmoid | 1/(1+e^-x) | (0, 1) | Binary output |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | (-1, 1) | Centered output |

## 📱 Mobile-Friendly Format

All cheat sheets are designed to be:
- **Scannable**: Key information highlighted
- **Concise**: Essential concepts only
- **Practical**: Ready-to-use code snippets
- **Visual**: Diagrams and flowcharts where helpful

## 🔄 How to Use

### During Learning
- Keep relevant cheat sheet open while studying
- Use as quick reference during coding
- Review before exams or interviews

### During Projects
- Copy-paste code snippets as starting points
- Check formula implementations
- Verify parameter ranges and defaults

### During Interviews
- Review key concepts quickly
- Refresh mathematical foundations
- Practice explaining concepts concisely

## 🤝 Contributing

Help improve these cheat sheets:

### Adding New Content
1. Follow the established format
2. Include both theory and practical examples
3. Keep explanations concise but complete
4. Add visual aids where helpful

### Updating Existing Sheets
1. Fix errors or outdated information
2. Add missing concepts
3. Improve code examples
4. Enhance visual presentation

### Quality Guidelines
- **Accuracy**: Double-check all formulas and code
- **Clarity**: Use simple, clear language
- **Completeness**: Cover essential concepts thoroughly
- **Consistency**: Follow formatting standards

## 📖 Recommended Usage Order

### For Beginners
1. Probability & Statistics
2. Linear Algebra
3. Neural Network Basics
4. TensorFlow/Keras or PyTorch

### For Intermediate Users
1. CNN Architectures
2. GANs Quick Reference
3. VAE Essentials
4. Evaluation Metrics

### For Advanced Users
1. Transformer Architecture
2. Diffusion Models
3. Training Diagnostics
4. Hyperparameter Tuning

## 🎯 Learning Tips

### Effective Usage
- **Print Key Sheets**: Keep physical copies handy
- **Customize**: Add your own notes and examples
- **Practice**: Implement examples from scratch
- **Teach**: Explain concepts to others using the sheets

### Memory Techniques
- **Spaced Repetition**: Review regularly
- **Active Recall**: Test yourself without looking
- **Visualization**: Draw diagrams and flowcharts
- **Application**: Use in real projects immediately

## 📚 Additional Resources

### Complementary Materials
- [Official Documentation Links](../Tools/documentation-links.md)
- [Research Paper Summaries](../Papers/paper-summaries.md)
- [Video Tutorial Playlists](../Tools/video-resources.md)

### Interactive Tools
- [Online Calculators](../Tools/online-calculators.md)
- [Visualization Tools](../Tools/visualization-tools.md)
- [Code Playgrounds](../Tools/code-playgrounds.md)

---

**Quick Access**: Bookmark this page and keep your most-used cheat sheets in browser tabs! 🔖

*These cheat sheets are living documents - they improve with your feedback and contributions!*