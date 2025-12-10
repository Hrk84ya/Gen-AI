# Transformer Architecture

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the transformer architecture and attention mechanisms
- Implement multi-head self-attention from scratch
- Master positional encoding and layer normalization
- Build encoder-decoder and encoder-only transformers
- Apply transformers to various tasks (NLP, vision, multimodal)
- Understand modern transformer variants and optimizations

## 🔄 What are Transformers?

Transformers are a neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output. They have revolutionized not just NLP, but the entire field of AI.

### Key Innovations:
- **Self-Attention**: Relate different positions in a sequence to compute representations
- **Parallelization**: No recurrence, enabling efficient training
- **Long-Range Dependencies**: Direct connections between distant positions
- **Scalability**: Performance improves with model and data size

### Impact:
- **NLP Revolution**: BERT, GPT, T5, ChatGPT
- **Computer Vision**: Vision Transformer (ViT), DETR
- **Multimodal AI**: CLIP, DALL-E, GPT-4
- **Scientific Computing**: AlphaFold, protein structure prediction

## 📐 Mathematical Foundation

### Self-Attention Mechanism
Given input sequence X, compute:
```
Q = XW_Q  (Queries)
K = XW_K  (Keys)  
V = XW_V  (Values)

Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## 📚 Module Contents

### 1. [Attention Mechanisms](./01_attention_mechanisms.ipynb)
- Scaled dot-product attention
- Multi-head attention implementation
- Attention visualization and interpretation
- Comparison with RNN/CNN approaches

### 2. [Transformer Building Blocks](./02_transformer_blocks.ipynb)
- Positional encoding strategies
- Layer normalization and residual connections
- Feed-forward networks
- Complete transformer layer implementation

### 3. [Encoder-Decoder Architecture](./03_encoder_decoder.ipynb)
- Full transformer implementation
- Training for sequence-to-sequence tasks
- Machine translation example
- Beam search and decoding strategies

### 4. [Encoder-Only Models](./04_encoder_only.ipynb)
- BERT-style architecture
- Masked language modeling
- Classification and representation learning
- Fine-tuning strategies

### 5. [Decoder-Only Models](./05_decoder_only.ipynb)
- GPT-style architecture
- Autoregressive generation
- Language modeling and text generation
- Scaling laws and emergent abilities

### 6. [Vision Transformers](./06_vision_transformers.ipynb)
- Patch embeddings for images
- ViT architecture and variants
- Comparison with CNNs
- Hybrid approaches

### 7. [Advanced Techniques](./07_advanced_techniques.ipynb)
- Efficient attention mechanisms
- Sparse attention patterns
- Model compression and distillation
- Latest research developments

## 🏗️ Core Components Implementation

### 1. **Scaled Dot-Product Attention**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot-product attention
    
    Args:
        query: [batch_size, seq_len, d_k]
        key: [batch_size, seq_len, d_k]
        value: [batch_size, seq_len, d_v]
        mask: [batch_size, seq_len, seq_len] or broadcastable
        dropout: dropout layer
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights
```

### 2. **Positional Encoding**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        
    def forward(self, x):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings
```

### 3. **Transformer Block**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_seq_length, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights
```

### 4. **Complete Transformer Model**
```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, 
            d_ff, max_seq_length, dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, 
            d_ff, max_seq_length, dropout
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        encoder_output, encoder_attention = self.encoder(src, src_mask)
        
        # Decode target sequence
        decoder_output, decoder_attention, cross_attention = self.decoder(
            tgt, encoder_output, tgt_mask, src_mask
        )
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output, {
            'encoder_attention': encoder_attention,
            'decoder_attention': decoder_attention,
            'cross_attention': cross_attention
        }

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_seq_length, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, tgt_mask, src_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        return x, self_attention_weights, cross_attention_weights

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Self-attention
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights
```

## 🎯 Training Strategies

### 1. **Learning Rate Scheduling**
```python
class TransformerLRScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

# Usage with PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=TransformerLRScheduler(d_model=512)
)
```

### 2. **Label Smoothing**
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # pred: [batch_size, seq_len, vocab_size]
        # target: [batch_size, seq_len]
        
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Mask ignored indices
        mask = (target != self.ignore_index)
        true_dist = true_dist * mask.unsqueeze(1).float()
        
        # Compute KL divergence
        return F.kl_div(F.log_softmax(pred, dim=1), true_dist, reduction='sum')
```

### 3. **Attention Visualization**
```python
def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """
    Visualize attention weights as a heatmap
    
    Args:
        attention_weights: [num_layers, batch_size, num_heads, seq_len, seq_len]
        tokens: List of token strings
        layer: Which layer to visualize
        head: Which attention head to visualize
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract attention for specific layer and head
    attn = attention_weights[layer][0, head].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, 
                cmap='Blues', cbar=True, square=True)
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def attention_rollout(attention_weights, discard_ratio=0.9):
    """
    Compute attention rollout to see information flow
    """
    result = torch.eye(attention_weights.size(-1))
    
    for attention in attention_weights:
        # Average over heads
        attention_heads_fused = attention.mean(dim=1)
        
        # Drop lowest attentions
        flat = attention_heads_fused.view(-1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
        flat[indices] = 0
        
        # Renormalize
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        
        # Add residual
        I = torch.eye(attention_heads_fused.size(-1))
        attention_heads_fused = (attention_heads_fused + I) / 2
        
        # Multiply
        result = torch.matmul(attention_heads_fused, result)
    
    return result
```

## 🔧 Optimization Techniques

### 1. **Efficient Attention Mechanisms**
```python
class LinearAttention(nn.Module):
    """Linear attention with O(n) complexity"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply feature map (e.g., ELU + 1)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Linear attention: O(n) complexity
        KV = torch.einsum('bshd,bshv->bhdv', K, V)
        Z = torch.einsum('bshd,bhdv->bshv', Q, KV)
        
        # Normalization
        normalizer = torch.einsum('bshd,bhd->bsh', Q, K.sum(dim=1))
        Z = Z / normalizer.unsqueeze(-1)
        
        # Reshape and project
        Z = Z.contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(Z)

class SparseAttention(nn.Module):
    """Sparse attention with configurable patterns"""
    def __init__(self, d_model, num_heads, sparsity_pattern='local'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.sparsity_pattern = sparsity_pattern
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        
    def create_sparse_mask(self, seq_len, pattern='local', window_size=128):
        mask = torch.zeros(seq_len, seq_len)
        
        if pattern == 'local':
            # Local attention window
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = 1
                
        elif pattern == 'strided':
            # Strided attention
            stride = window_size
            for i in range(seq_len):
                mask[i, i::stride] = 1
                
        elif pattern == 'global':
            # Global tokens attend to everything
            mask[:window_size, :] = 1
            mask[:, :window_size] = 1
            
        return mask
        
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.create_sparse_mask(seq_len, self.sparsity_pattern)
        mask = mask.to(x.device)
        
        return self.attention(x, x, x, mask)
```

### 2. **Model Compression**
```python
class DistilledTransformer(nn.Module):
    """Smaller transformer trained with knowledge distillation"""
    def __init__(self, teacher_model, student_config):
        super().__init__()
        self.teacher = teacher_model
        self.student = Transformer(**student_config)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def distillation_loss(self, student_logits, teacher_logits, labels, 
                         temperature=3.0, alpha=0.7):
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        
        # Distillation loss
        distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        distill_loss *= (temperature ** 2)
        
        # Student loss
        student_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                     labels.view(-1))
        
        return alpha * distill_loss + (1 - alpha) * student_loss
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_output, _ = self.teacher(src, tgt, src_mask, tgt_mask)
            
        # Student forward
        student_output, attention_weights = self.student(src, tgt, src_mask, tgt_mask)
        
        return student_output, teacher_output, attention_weights
```

## 🎨 Applications

### 1. **Text Classification (BERT-style)**
```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=768, num_heads=12, 
                 num_layers=12, max_seq_length=512):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, 
            d_model * 4, max_seq_length
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Encode sequence
        encoded, _ = self.encoder(input_ids, attention_mask)
        
        # Use [CLS] token representation (first token)
        cls_representation = encoded[:, 0]
        
        # Classify
        logits = self.classifier(cls_representation)
        
        return logits
```

### 2. **Language Generation (GPT-style)**
```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 max_seq_length=1024):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = LearnedPositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        # Create causal mask
        seq_len = input_ids.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_ids.device)
        
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask.unsqueeze(1)
        
        # Embedding
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Transformer layers
        for layer in self.layers:
            x, _ = layer(x, causal_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """Generate text autoregressively"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
```

## 📊 Evaluation and Analysis

### 1. **Attention Pattern Analysis**
```python
def analyze_attention_patterns(model, dataloader, device):
    """Analyze what the model attends to"""
    model.eval()
    
    attention_stats = {
        'head_entropy': [],
        'layer_entropy': [],
        'attention_distance': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            _, attention_weights = model.encoder(input_ids)
            
            for layer_idx, layer_attn in enumerate(attention_weights):
                # layer_attn: [batch_size, num_heads, seq_len, seq_len]
                
                for head_idx in range(layer_attn.size(1)):
                    head_attn = layer_attn[:, head_idx]  # [batch_size, seq_len, seq_len]
                    
                    # Compute entropy (how focused the attention is)
                    entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-8), dim=-1)
                    attention_stats['head_entropy'].append(entropy.mean().item())
                    
                    # Compute average attention distance
                    seq_len = head_attn.size(-1)
                    positions = torch.arange(seq_len, device=device).float()
                    
                    # Weighted average of attended positions
                    attended_positions = torch.sum(head_attn * positions.unsqueeze(0).unsqueeze(0), dim=-1)
                    query_positions = positions.unsqueeze(0).unsqueeze(0)
                    
                    distances = torch.abs(attended_positions - query_positions)
                    attention_stats['attention_distance'].append(distances.mean().item())
            
            break  # Analyze just one batch for demo
    
    return attention_stats

def plot_attention_heads(attention_weights, layer_names=None):
    """Plot attention patterns across heads and layers"""
    import matplotlib.pyplot as plt
    
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].size(1)
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 2, num_layers * 2))
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        for head_idx in range(num_heads):
            # Average over batch dimension
            head_attn = layer_attn[0, head_idx].cpu().numpy()
            
            ax = axes[layer_idx, head_idx] if num_layers > 1 else axes[head_idx]
            im = ax.imshow(head_attn, cmap='Blues')
            ax.set_title(f'L{layer_idx}H{head_idx}')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
```

## 🎓 Learning Path

### Week 1: Attention Fundamentals
- Understand attention mechanisms
- Implement scaled dot-product attention
- Build multi-head attention

### Week 2: Transformer Architecture
- Learn positional encoding
- Implement transformer blocks
- Build complete encoder-decoder

### Week 3: Model Variants
- Encoder-only models (BERT-style)
- Decoder-only models (GPT-style)
- Vision transformers

### Week 4: Advanced Techniques
- Efficient attention mechanisms
- Model compression and distillation
- Latest research developments

## 📚 Additional Resources

### Foundational Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Unsupervised Multitask Learners" (GPT-2)
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)

### Advanced Papers
- "Reformer: The Efficient Transformer" (Kitaev et al., 2020)
- "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
- "Switch Transformer: Scaling to Trillion Parameter Models" (Fedus et al., 2021)

### Practical Resources
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers Library](https://huggingface.co/transformers/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

---

**Next Module**: [Large Language Models](../05-LLMs/) →

*Ready to master the architecture that changed AI forever? Let's build transformers from the ground up! 🔄✨*