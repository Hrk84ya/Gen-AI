# 🔄 Transformers Cheat Sheet

## 🎯 Quick Overview

**Transformers** are neural network architectures that use self-attention mechanisms to process sequences. They're the foundation of modern NLP and generative AI.

## 🏗️ Architecture Components

### 1. Self-Attention Mechanism
```python
# Simplified attention calculation
def attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output
```

### 2. Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

### 3. Position Encoding
```python
def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe
```

## 🔧 Common Implementations

### Using Hugging Face Transformers
```python
from transformers import AutoModel, AutoTokenizer

# Load pre-trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize and encode
text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### Text Generation with GPT
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate text
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

## 📊 Model Variants

| Model | Use Case | Key Features |
|-------|----------|--------------|
| **BERT** | Understanding | Bidirectional, masked language modeling |
| **GPT** | Generation | Autoregressive, causal attention |
| **T5** | Text-to-text | Encoder-decoder, unified framework |
| **RoBERTa** | Understanding | Optimized BERT training |
| **ELECTRA** | Efficiency | Replaced token detection |

## ⚙️ Key Hyperparameters

```python
# Common transformer hyperparameters
config = {
    "vocab_size": 50257,
    "n_positions": 1024,      # Max sequence length
    "n_ctx": 1024,           # Context size
    "n_embd": 768,           # Embedding dimension
    "n_layer": 12,           # Number of layers
    "n_head": 12,            # Number of attention heads
    "dropout": 0.1,
    "attention_dropout": 0.1
}
```

## 🎯 Training Tips

### 1. Learning Rate Scheduling
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

### 2. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(**inputs)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🔍 Attention Patterns

### Self-Attention Types
- **Causal**: Can only attend to previous tokens (GPT)
- **Bidirectional**: Can attend to all tokens (BERT)
- **Masked**: Specific tokens are masked during attention

### Attention Visualization
```python
# Extract attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # List of attention matrices

# Visualize attention
import matplotlib.pyplot as plt
import seaborn as sns

attention_matrix = attentions[0][0][0].numpy()  # First layer, first head
sns.heatmap(attention_matrix, cmap='Blues')
plt.title('Attention Weights')
plt.show()
```

## 🚀 Fine-tuning Strategies

### 1. Full Fine-tuning
```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Train normally
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

### 2. Frozen Feature Extraction
```python
# Freeze base model
for param in model.base_model.parameters():
    param.requires_grad = False

# Only train classifier head
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
```

### 3. Layer-wise Learning Rates
```python
# Different learning rates for different layers
param_groups = [
    {"params": model.embeddings.parameters(), "lr": 1e-5},
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4}
]
optimizer = torch.optim.AdamW(param_groups)
```

## 📈 Performance Optimization

### Memory Optimization
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller batch sizes with gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Speed Optimization
```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use DataLoader optimizations
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## 🎯 Common Tasks & Code

### Text Classification
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Question Answering
```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Inference
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits
```

### Text Summarization
```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Generate summary
input_text = "summarize: " + long_text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(
    inputs.input_ids,
    max_length=150,
    min_length=30,
    length_penalty=2.0,
    num_beams=4
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

## 🐛 Common Issues & Solutions

### Issue: Out of Memory
```python
# Solutions:
# 1. Reduce batch size
# 2. Use gradient accumulation
# 3. Enable gradient checkpointing
# 4. Use mixed precision training
```

### Issue: Slow Training
```python
# Solutions:
# 1. Use DataLoader with multiple workers
# 2. Enable model compilation
# 3. Use appropriate device placement
# 4. Optimize data preprocessing
```

### Issue: Poor Performance
```python
# Solutions:
# 1. Check learning rate (try 2e-5 to 5e-5)
# 2. Verify data preprocessing
# 3. Use appropriate model size
# 4. Add regularization (dropout, weight decay)
```

## 📚 Key Papers & Resources

- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT** (Devlin et al., 2018)
- **GPT** series (Radford et al.)
- **T5** (Raffel et al., 2019)

## 🔗 Useful Links

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)

---
*Keep this cheat sheet handy while working with transformers! 🚀*