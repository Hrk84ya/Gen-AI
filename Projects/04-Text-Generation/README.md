# 📝 Text Generation Project

## Project Overview
Build advanced text generation systems using state-of-the-art language models, from character-level RNNs to transformer-based models like GPT. This project covers the complete pipeline of text generation including data preprocessing, model training, fine-tuning, and deployment for various applications.

## 🎯 Learning Objectives
- Understand different text generation approaches and architectures
- Implement RNN, LSTM, and Transformer-based language models
- Master techniques for training stable and coherent text generators
- Fine-tune pre-trained language models for specific domains
- Deploy text generation systems for real-world applications

## 🏗️ Project Structure
```
04-Text-Generation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── books/
│   │   ├── articles/
│   │   └── dialogues/
│   ├── processed/
│   └── tokenized/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── char_rnn.py
│   │   ├── word_lstm.py
│   │   ├── transformer.py
│   │   ├── gpt_model.py
│   │   └── fine_tuning.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── schedulers.py
│   ├── generation/
│   │   ├── sampling.py
│   │   ├── beam_search.py
│   │   └── nucleus_sampling.py
│   ├── utils/
│   │   ├── tokenizer.py
│   │   ├── data_loader.py
│   │   └── text_utils.py
│   └── evaluation/
│       ├── perplexity.py
│       ├── bleu_score.py
│       └── human_eval.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_char_level_generation.ipynb
│   ├── 03_word_level_lstm.ipynb
│   ├── 04_transformer_training.ipynb
│   ├── 05_gpt_fine_tuning.ipynb
│   └── 06_evaluation_metrics.ipynb
├── configs/
│   ├── char_rnn.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   └── gpt_finetune.yaml
├── experiments/
│   ├── char_rnn/
│   ├── lstm/
│   ├── transformer/
│   └── gpt_finetune/
├── web_app/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── generate.html
│   └── static/
│       ├── css/
│       └── js/
├── api/
│   ├── fastapi_server.py
│   ├── models.py
│   └── endpoints.py
└── deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    └── kubernetes/
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Navigate to project
cd Projects/04-Text-Generation

# Create virtual environment
python -m venv textgen_env
source textgen_env/bin/activate  # On Windows: textgen_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download sample datasets
python src/utils/data_loader.py --download --dataset shakespeare
```

### 2. Train Your First Model
```bash
# Character-level RNN on Shakespeare
python src/training/trainer.py --config configs/char_rnn.yaml --dataset shakespeare

# Word-level LSTM on news articles
python src/training/trainer.py --config configs/lstm.yaml --dataset news

# Fine-tune GPT-2 on custom data
python src/training/trainer.py --config configs/gpt_finetune.yaml --dataset custom
```

### 3. Generate Text
```bash
# Generate text with trained model
python src/generate.py --model experiments/char_rnn/best_model.pth --prompt "To be or not to be" --length 200

# Interactive generation
python src/interactive_generation.py --model experiments/lstm/best_model.pth

# Web interface
python web_app/app.py
```

## 📊 Supported Datasets

### Text Corpora
- **Shakespeare**: Complete works of William Shakespeare
- **Gutenberg**: Public domain books from Project Gutenberg
- **News Articles**: Reuters, CNN, BBC news articles
- **Wikipedia**: Wikipedia article excerpts
- **Reddit Comments**: Conversational text from Reddit
- **Code**: Programming code from GitHub repositories
- **Custom**: Your own text datasets

### Dataset Statistics
| Dataset | Size | Vocabulary | Domain | Complexity |
|---------|------|------------|---------|------------|
| Shakespeare | 1.1MB | 65K words | Literature | High |
| News | 50MB | 200K words | Journalism | Medium |
| Wikipedia | 100MB | 500K words | Encyclopedia | Medium |
| Reddit | 200MB | 1M words | Conversational | Low |
| Code | 500MB | 100K tokens | Programming | High |

## 🧠 Model Architectures

### 1. Character-Level RNN
**Architecture**: Simple recurrent neural network operating on characters
```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.output(output)
        return output, hidden
```

**Use Cases**:
- Learning sequence modeling fundamentals
- Small datasets with limited vocabulary
- Character-level tasks (poetry, code generation)

### 2. Word-Level LSTM
**Architecture**: Long Short-Term Memory networks with word embeddings
```python
class WordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.output(output)
        return output, hidden
```

**Features**:
- Better long-term dependencies than RNN
- Word-level understanding
- Suitable for medium-length text generation

### 3. Transformer Language Model
**Architecture**: Self-attention based transformer decoder
```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, 
                                                  dim_feedforward=4*d_model)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        output = self.transformer(x, mask=mask)
        return self.output(output)
```

**Advantages**:
- Parallel training (faster than RNNs)
- Better long-range dependencies
- State-of-the-art performance

### 4. GPT-Style Model
**Architecture**: Generative Pre-trained Transformer
```python
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

**Features**:
- Pre-training + fine-tuning paradigm
- Scalable to very large models
- Excellent few-shot learning capabilities

## 🔧 Text Generation Techniques

### 1. Sampling Strategies
```python
def temperature_sampling(logits, temperature=1.0):
    """Apply temperature scaling to logits"""
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, 1)

def top_k_sampling(logits, k=50):
    """Sample from top-k most likely tokens"""
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(probs, 1)
    return top_k_indices.gather(-1, sampled_index)

def nucleus_sampling(logits, p=0.9):
    """Sample from nucleus (top-p) of probability mass"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Set logits to -inf for removed tokens
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # Sample from remaining distribution
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_sorted_index = torch.multinomial(probs, 1)
    sampled_index = sorted_indices.gather(-1, sampled_sorted_index)
    
    return sampled_index
```

### 2. Beam Search
```python
class BeamSearch:
    def __init__(self, model, beam_size=5, max_length=100):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
    
    def search(self, start_tokens, end_token=None):
        """Perform beam search decoding"""
        batch_size = start_tokens.size(0)
        
        # Initialize beams
        beams = [(start_tokens, 0.0)]  # (sequence, score)
        
        for step in range(self.max_length):
            candidates = []
            
            for sequence, score in beams:
                if sequence[0, -1] == end_token:
                    candidates.append((sequence, score))
                    continue
                
                # Get next token probabilities
                with torch.no_grad():
                    logits = self.model(sequence)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Get top-k candidates
                top_log_probs, top_indices = torch.topk(log_probs, self.beam_size)
                
                for i in range(self.beam_size):
                    new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_sequence = torch.cat([sequence, new_token], dim=1)
                    new_score = score + top_log_probs[0, i].item()
                    candidates.append((new_sequence, new_score))
            
            # Select top beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_size]
        
        return beams[0][0]  # Return best sequence
```

### 3. Controllable Generation
```python
class ControllableGenerator:
    def __init__(self, model, classifier):
        self.model = model
        self.classifier = classifier  # For attribute control
    
    def generate_with_attribute(self, prompt, target_attribute, strength=1.0):
        """Generate text with specific attributes (sentiment, style, etc.)"""
        sequence = self.tokenize(prompt)
        
        for _ in range(self.max_length):
            # Get model predictions
            logits = self.model(sequence)
            
            # Get classifier gradients for attribute control
            sequence.requires_grad_(True)
            attr_logits = self.classifier(sequence)
            attr_loss = F.cross_entropy(attr_logits, target_attribute)
            
            # Compute gradients
            attr_grad = torch.autograd.grad(attr_loss, sequence)[0]
            
            # Modify logits based on attribute gradients
            modified_logits = logits[:, -1, :] + strength * attr_grad[:, -1, :]
            
            # Sample next token
            next_token = self.sample(modified_logits)
            sequence = torch.cat([sequence, next_token.unsqueeze(1)], dim=1)
        
        return self.detokenize(sequence)
    
    def generate_with_style_transfer(self, content, target_style):
        """Transfer style while preserving content"""
        # Extract content representation
        content_repr = self.extract_content(content)
        
        # Extract target style representation
        style_repr = self.extract_style(target_style)
        
        # Generate text with combined representations
        return self.generate_from_representations(content_repr, style_repr)
```

## 📈 Training Strategies

### 1. Curriculum Learning
```python
class CurriculumTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.current_seq_length = 32  # Start with short sequences
        self.max_seq_length = 512
    
    def train_epoch(self, epoch):
        # Gradually increase sequence length
        if epoch % 10 == 0 and self.current_seq_length < self.max_seq_length:
            self.current_seq_length = min(
                self.current_seq_length * 2, 
                self.max_seq_length
            )
            print(f"Increased sequence length to {self.current_seq_length}")
        
        # Train with current sequence length
        for batch in self.data_loader:
            # Truncate sequences to current length
            inputs = batch[:, :self.current_seq_length]
            targets = batch[:, 1:self.current_seq_length+1]
            
            # Forward pass and training step
            loss = self.train_step(inputs, targets)
```

### 2. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(inputs)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### 3. Gradient Accumulation
```python
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
    
    def train_epoch(self, data_loader):
        self.model.train()
        accumulated_loss = 0
        
        for i, batch in enumerate(data_loader):
            inputs, targets = batch
            
            # Forward pass
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                                 targets.view(-1))
            
            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return accumulated_loss
```

## 📊 Evaluation Metrics

### 1. Perplexity
```python
def calculate_perplexity(model, data_loader, device):
    """Calculate perplexity on validation data"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

### 2. BLEU Score
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def calculate_bleu_score(generated_texts, reference_texts):
    """Calculate BLEU score for generated text"""
    bleu_scores = []
    
    for generated, reference in zip(generated_texts, reference_texts):
        # Tokenize texts
        generated_tokens = generated.split()
        reference_tokens = [reference.split()]  # List of reference lists
        
        # Calculate BLEU score
        score = sentence_bleu(reference_tokens, generated_tokens)
        bleu_scores.append(score)
    
    return sum(bleu_scores) / len(bleu_scores)
```

### 3. Diversity Metrics
```python
def calculate_diversity_metrics(generated_texts):
    """Calculate diversity metrics for generated text"""
    all_tokens = []
    all_bigrams = []
    all_trigrams = []
    
    for text in generated_texts:
        tokens = text.split()
        all_tokens.extend(tokens)
        
        # Generate n-grams
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        
        all_bigrams.extend(bigrams)
        all_trigrams.extend(trigrams)
    
    # Calculate diversity as ratio of unique n-grams
    metrics = {
        'token_diversity': len(set(all_tokens)) / len(all_tokens),
        'bigram_diversity': len(set(all_bigrams)) / len(all_bigrams),
        'trigram_diversity': len(set(all_trigrams)) / len(all_trigrams)
    }
    
    return metrics
```

## 🎨 Applications

### 1. Creative Writing Assistant
```python
class CreativeWritingAssistant:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.genres = ['fantasy', 'sci-fi', 'mystery', 'romance', 'horror']
    
    def continue_story(self, prompt, genre='general', length=200):
        """Continue a story in a specific genre"""
        # Add genre conditioning
        conditioned_prompt = f"[{genre.upper()}] {prompt}"
        
        # Generate continuation
        generated = self.model.generate(
            conditioned_prompt,
            max_length=length,
            temperature=0.8,
            top_p=0.9
        )
        
        return generated
    
    def suggest_plot_twist(self, story_context):
        """Suggest plot twists based on story context"""
        twist_prompt = f"Given this story: {story_context}\nA surprising plot twist would be:"
        
        twists = []
        for _ in range(3):  # Generate multiple options
            twist = self.model.generate(
                twist_prompt,
                max_length=100,
                temperature=1.0,
                do_sample=True
            )
            twists.append(twist)
        
        return twists
    
    def generate_character_dialogue(self, character_description, situation):
        """Generate character dialogue for specific situations"""
        dialogue_prompt = f"""
        Character: {character_description}
        Situation: {situation}
        Character says: "
        """
        
        dialogue = self.model.generate(
            dialogue_prompt,
            max_length=150,
            temperature=0.7,
            stop_tokens=['"']
        )
        
        return dialogue
```

### 2. Code Generation
```python
class CodeGenerator:
    def __init__(self, model_path, language='python'):
        self.model = self.load_model(model_path)
        self.language = language
    
    def generate_function(self, description, function_name=None):
        """Generate code function from description"""
        if function_name:
            prompt = f"# {description}\ndef {function_name}("
        else:
            prompt = f"# {description}\ndef "
        
        code = self.model.generate(
            prompt,
            max_length=300,
            temperature=0.2,  # Lower temperature for more deterministic code
            stop_tokens=['\n\n', '# ']
        )
        
        return self.format_code(code)
    
    def complete_code(self, partial_code):
        """Complete partial code"""
        completion = self.model.generate(
            partial_code,
            max_length=200,
            temperature=0.3
        )
        
        return completion
    
    def explain_code(self, code_snippet):
        """Generate explanation for code"""
        explanation_prompt = f"""
        Code:
        {code_snippet}
        
        Explanation:
        This code
        """
        
        explanation = self.model.generate(
            explanation_prompt,
            max_length=200,
            temperature=0.5
        )
        
        return explanation
    
    def format_code(self, code):
        """Format generated code"""
        # Basic code formatting
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                if stripped.endswith(':'):
                    formatted_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                else:
                    formatted_lines.append('    ' * indent_level + stripped)
            else:
                formatted_lines.append('')
        
        return '\n'.join(formatted_lines)
```

### 3. Chatbot Response Generation
```python
class ConversationalAgent:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.conversation_history = []
        self.personality = "helpful and friendly"
    
    def generate_response(self, user_input, context_length=5):
        """Generate conversational response"""
        # Build conversation context
        recent_history = self.conversation_history[-context_length:]
        context = self.build_context(recent_history)
        
        # Create prompt with personality and context
        prompt = f"""
        Personality: {self.personality}
        Conversation:
        {context}
        Human: {user_input}
        Assistant:
        """
        
        response = self.model.generate(
            prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            stop_tokens=['Human:', 'Assistant:']
        )
        
        # Update conversation history
        self.conversation_history.append(('Human', user_input))
        self.conversation_history.append(('Assistant', response))
        
        return response
    
    def build_context(self, history):
        """Build conversation context from history"""
        context_lines = []
        for speaker, message in history:
            context_lines.append(f"{speaker}: {message}")
        return '\n'.join(context_lines)
    
    def set_personality(self, personality_description):
        """Update agent personality"""
        self.personality = personality_description
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
```

## 🌐 Web Application

### Flask Web Interface
```python
from flask import Flask, render_template, request, jsonify, stream_template
import json

app = Flask(__name__)
text_generator = None  # Initialize with your model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    length = data.get('length', 100)
    temperature = data.get('temperature', 0.8)
    top_p = data.get('top_p', 0.9)
    
    try:
        generated_text = text_generator.generate(
            prompt=prompt,
            max_length=length,
            temperature=temperature,
            top_p=top_p
        )
        
        return jsonify({
            'success': True,
            'generated_text': generated_text,
            'prompt': prompt
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stream_generate', methods=['POST'])
def stream_generate():
    """Stream generated text token by token"""
    data = request.json
    prompt = data.get('prompt', '')
    
    def generate_stream():
        for token in text_generator.generate_stream(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return Response(generate_stream(), mimetype='text/plain')

@app.route('/fine_tune', methods=['POST'])
def fine_tune_model():
    """Fine-tune model on user data"""
    data = request.json
    training_text = data.get('training_text', '')
    epochs = data.get('epochs', 3)
    
    # Start fine-tuning in background
    task_id = start_fine_tuning_task(training_text, epochs)
    
    return jsonify({
        'task_id': task_id,
        'status': 'started'
    })

@app.route('/fine_tune_status/<task_id>')
def fine_tune_status(task_id):
    """Check fine-tuning status"""
    status = get_fine_tuning_status(task_id)
    return jsonify(status)
```

## 📊 Performance Optimization

### 1. Model Quantization
```python
import torch.quantization as quantization

def quantize_model(model, calibration_data):
    """Quantize model for faster inference"""
    # Prepare model for quantization
    model.eval()
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    # Convert to quantized model
    quantized_model = quantization.convert(model, inplace=False)
    
    return quantized_model
```

### 2. Model Distillation
```python
class ModelDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def distillation_loss(self, student_logits, teacher_logits, targets, alpha=0.7):
        """Compute distillation loss"""
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss
        distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        distill_loss *= (self.temperature ** 2)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
        
        return total_loss
    
    def train_student(self, data_loader, optimizer, epochs=10):
        """Train student model with distillation"""
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(epochs):
            for batch in data_loader:
                inputs, targets = batch
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                # Get student predictions
                student_logits = self.student(inputs)
                
                # Compute distillation loss
                loss = self.distillation_loss(
                    student_logits, teacher_logits, targets
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 3. Caching and Optimization
```python
class OptimizedGenerator:
    def __init__(self, model):
        self.model = model
        self.kv_cache = {}  # Key-value cache for attention
        self.compiled_model = None
    
    def compile_model(self):
        """Compile model for faster inference"""
        if hasattr(torch, 'compile'):
            self.compiled_model = torch.compile(self.model)
        else:
            self.compiled_model = self.model
    
    def generate_with_cache(self, prompt, max_length=100):
        """Generate text with KV caching"""
        model = self.compiled_model or self.model
        
        tokens = self.tokenize(prompt)
        generated = tokens.clone()
        
        for _ in range(max_length):
            # Use cached computations when possible
            if len(generated) > len(tokens):
                # Only compute for new token
                input_ids = generated[-1:].unsqueeze(0)
                past_key_values = self.kv_cache.get('past_key_values')
            else:
                input_ids = generated.unsqueeze(0)
                past_key_values = None
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            # Update cache
            self.kv_cache['past_key_values'] = outputs.past_key_values
            
            # Sample next token
            next_token = self.sample(outputs.logits[:, -1, :])
            generated = torch.cat([generated, next_token])
        
        return self.detokenize(generated)
```

## 📚 Learning Resources

### Research Papers
- **Attention Is All You Need**: Original Transformer paper
- **Language Models are Unsupervised Multitask Learners**: GPT-2 paper
- **BERT**: Bidirectional Encoder Representations from Transformers
- **T5**: Text-to-Text Transfer Transformer
- **PaLM**: Scaling Language Modeling with Pathways

### Books and Courses
- "Natural Language Processing with Python" by Steven Bird
- "Speech and Language Processing" by Dan Jurafsky
- CS224N: Natural Language Processing with Deep Learning (Stanford)
- Hugging Face NLP Course

### Online Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI GPT Models](https://openai.com/research/)
- [Papers With Code - Language Modeling](https://paperswithcode.com/task/language-modelling)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## 🚀 Future Enhancements

### Planned Features
1. **Multimodal Generation**: Text + image generation
2. **Real-time Collaboration**: Multiple users editing together
3. **Advanced Fine-tuning**: LoRA, QLoRA, and other PEFT methods
4. **Retrieval-Augmented Generation**: RAG for factual accuracy
5. **Constitutional AI**: Value-aligned text generation

### Research Directions
- **Efficiency**: Smaller, faster models with maintained quality
- **Controllability**: Better control over generation attributes
- **Factuality**: Reducing hallucinations and improving accuracy
- **Multilinguality**: Better cross-lingual generation capabilities
- **Safety**: Detecting and preventing harmful content generation

---

**Next Steps**: After mastering text generation, explore the [Multimodal App Project](../05-Multimodal-App/) to learn about combining text, image, and audio modalities in AI applications.