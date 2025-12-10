# Large Language Models (LLMs)

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the architecture and training of large language models
- Learn pre-training strategies and scaling laws
- Master fine-tuning techniques (supervised, RLHF, instruction tuning)
- Implement prompt engineering and in-context learning
- Build and deploy LLM applications
- Understand the capabilities and limitations of modern LLMs

## 🧠 What are Large Language Models?

Large Language Models are transformer-based neural networks trained on vast amounts of text data to understand and generate human-like text. They exhibit emergent abilities that arise from scale, including few-shot learning, reasoning, and code generation.

### Key Characteristics:
- **Scale**: Billions to trillions of parameters
- **Pre-training**: Unsupervised learning on massive text corpora
- **Emergent Abilities**: Capabilities that emerge at sufficient scale
- **Few-shot Learning**: Learning new tasks from just a few examples
- **Generalization**: Broad applicability across domains and tasks

### Evolution Timeline:
```
2018: BERT (340M parameters) - Bidirectional encoder
2019: GPT-2 (1.5B parameters) - Autoregressive decoder
2020: GPT-3 (175B parameters) - Few-shot learning breakthrough
2021: PaLM (540B parameters) - Scaling laws validation
2022: ChatGPT - RLHF and conversational AI
2023: GPT-4 - Multimodal capabilities
2024: Gemini, Claude - Advanced reasoning
```

## 📚 Module Contents

### 1. [LLM Fundamentals](./01_llm_fundamentals.ipynb)
- Architecture overview and design choices
- Scaling laws and emergent abilities
- Training data and preprocessing
- Tokenization strategies

### 2. [Pre-training Strategies](./02_pretraining.ipynb)
- Language modeling objectives
- Training infrastructure and optimization
- Data parallelism and model parallelism
- Gradient accumulation and mixed precision

### 3. [Fine-tuning Techniques](./03_finetuning.ipynb)
- Supervised fine-tuning (SFT)
- Parameter-efficient fine-tuning (LoRA, Adapters)
- Instruction tuning and alignment
- Domain adaptation strategies

### 4. [RLHF and Alignment](./04_rlhf_alignment.ipynb)
- Reinforcement Learning from Human Feedback
- Reward modeling and preference learning
- PPO training for language models
- Constitutional AI and safety measures

### 5. [Prompt Engineering](./05_prompt_engineering.ipynb)
- Prompt design principles
- In-context learning and few-shot prompting
- Chain-of-thought reasoning
- Advanced prompting techniques

### 6. [LLM Applications](./06_llm_applications.ipynb)
- Text generation and completion
- Question answering and reasoning
- Code generation and debugging
- Creative writing and content creation

### 7. [Deployment and Optimization](./07_deployment_optimization.ipynb)
- Model serving and inference optimization
- Quantization and pruning
- Distributed inference
- API design and scaling

## 🏗️ LLM Architecture Components

### 1. **GPT-style Architecture**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTConfig:
    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, 
                 n_layer=12, n_head=12, n_inner=None, activation_function="gelu_new",
                 resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-5):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon

class GPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                           .view(1, 1, config.n_positions, config.n_positions))
        
    def forward(self, x, attention_mask=None, use_cache=False, past_key_value=None):
        B, T, C = x.size()
        
        # Calculate QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle past key-values for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        present = (k, v) if use_cache else None
        
        # Attention computation
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        seq_len = k.size(-2)
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present

class GPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # Activation function
        if config.activation_function == "gelu_new":
            self.act = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        else:
            self.act = F.gelu
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPTMLP(config)
    
    def forward(self, x, attention_mask=None, use_cache=False, past_key_value=None):
        # Pre-norm architecture
        attn_output, present = self.attn(
            self.ln_1(x), 
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        x = x + attn_output
        
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output
        
        return x, present

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, 
                use_cache=False, past_key_values=None):
        
        batch_size, seq_len = input_ids.shape
        
        # Position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
            position_ids = torch.arange(past_length, seq_len + past_length, 
                                      dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        hidden_states = self.drop(token_embeddings + position_embeddings)
        
        # Transformer blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            
            if use_cache:
                presents.append(present)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        return {
            'logits': lm_logits,
            'past_key_values': presents,
            'hidden_states': hidden_states
        }
```

### 2. **Scaling Laws Implementation**
```python
class ScalingLaws:
    """Implementation of scaling laws for language models"""
    
    @staticmethod
    def compute_loss(N, D, C):
        """
        Compute expected loss based on scaling laws
        N: Number of parameters
        D: Dataset size (tokens)
        C: Compute budget (FLOPs)
        """
        # Chinchilla scaling law coefficients (approximate)
        A = 406.4
        B = 410.7
        alpha = 0.34
        beta = 0.28
        
        # Loss = A / N^alpha + B / D^beta
        return A / (N ** alpha) + B / (D ** beta)
    
    @staticmethod
    def optimal_allocation(compute_budget):
        """
        Compute optimal model size and training tokens for given compute
        """
        # Chinchilla optimal ratios
        # For compute C, optimal N ∝ C^0.50, optimal D ∝ C^0.50
        
        # Approximate coefficients
        N_coeff = 1.3e-9  # Parameters per FLOP
        D_coeff = 1.8e-9  # Tokens per FLOP
        
        optimal_N = (compute_budget * N_coeff) ** 0.5
        optimal_D = (compute_budget * D_coeff) ** 0.5
        
        return optimal_N, optimal_D
    
    @staticmethod
    def emergent_abilities_threshold(task_type):
        """
        Approximate parameter thresholds for emergent abilities
        """
        thresholds = {
            'few_shot_learning': 1e9,      # ~1B parameters
            'chain_of_thought': 1e11,      # ~100B parameters
            'code_generation': 1e10,       # ~10B parameters
            'mathematical_reasoning': 5e10, # ~50B parameters
            'instruction_following': 1e9,   # ~1B parameters (with fine-tuning)
        }
        
        return thresholds.get(task_type, 1e12)

# Example usage
compute_budget = 1e23  # FLOPs
optimal_params, optimal_tokens = ScalingLaws.optimal_allocation(compute_budget)
print(f"Optimal allocation: {optimal_params:.1e} parameters, {optimal_tokens:.1e} tokens")
```

### 3. **Efficient Training Utilities**
```python
class LLMTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def compute_loss(self, batch):
        """Compute language modeling loss"""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Shift for causal LM
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        # Forward pass
        outputs = self.model(inputs, attention_mask=attention_mask[:, :-1] if attention_mask is not None else None)
        logits = outputs['logits']
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        return loss, logits
    
    def train_step(self, batch, optimizer, scaler=None):
        """Single training step with mixed precision"""
        self.model.train()
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss, _ = self.compute_loss(batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            loss, _ = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        return loss.item()
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Generate text from prompt"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
```

## 🎯 Fine-tuning Strategies

### 1. **Parameter-Efficient Fine-tuning (LoRA)**
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        # LoRA forward: x @ (A^T @ B^T) * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return lora_output

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features, 
            original_layer.out_features, 
            rank, alpha, dropout
        )
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        return original_output + lora_output

def apply_lora_to_model(model, target_modules=['c_attn', 'c_proj'], rank=16, alpha=32):
    """Apply LoRA to specified modules in the model"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent_module = model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent_module, child_name, lora_layer)
    
    return model
```

### 2. **Instruction Tuning**
```python
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format instruction
        if 'input' in item and item['input']:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        
        # Add response
        full_text = prompt + item['output']
        
        # Tokenize
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (mask prompt tokens)
        labels = tokens['input_ids'].clone()
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        labels[:, :len(prompt_tokens[0])] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def instruction_tuning_loss(logits, labels):
    """Compute loss only on response tokens"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    
    return loss
```

### 3. **RLHF Implementation**
```python
class RewardModel(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(config.n_embd, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs['hidden_states']
        
        # Use last token for reward prediction
        rewards = self.reward_head(hidden_states[:, -1])
        return rewards

class PPOTrainer:
    def __init__(self, policy_model, reward_model, ref_model, config):
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_rewards(self, queries, responses):
        """Compute rewards for query-response pairs"""
        rewards = []
        
        for query, response in zip(queries, responses):
            # Combine query and response
            full_text = query + response
            input_ids = self.tokenizer.encode(full_text, return_tensors='pt')
            
            # Get reward
            with torch.no_grad():
                reward = self.reward_model(input_ids).item()
            
            rewards.append(reward)
        
        return torch.tensor(rewards)
    
    def compute_kl_penalty(self, logprobs, ref_logprobs):
        """Compute KL divergence penalty"""
        kl = logprobs - ref_logprobs
        return kl
    
    def ppo_step(self, queries, responses, old_logprobs, rewards, advantages):
        """Single PPO update step"""
        # Forward pass through policy
        policy_outputs = self.policy(responses)
        logprobs = F.log_softmax(policy_outputs['logits'], dim=-1)
        
        # Compute probability ratios
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Clipped surrogate objective
        clip_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
        policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        # Value loss (if using critic)
        value_loss = 0  # Simplified
        
        # Total loss
        total_loss = policy_loss + self.config.vf_coef * value_loss
        
        return total_loss, policy_loss, value_loss
```

## 🎨 Advanced Applications

### 1. **Chain-of-Thought Reasoning**
```python
class ChainOfThoughtPrompt:
    def __init__(self):
        self.examples = [
            {
                "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                "reasoning": "Roger started with 5 tennis balls. 2 cans of 3 tennis balls each is 2 × 3 = 6 tennis balls. 5 + 6 = 11.",
                "answer": "11"
            },
            {
                "question": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
                "reasoning": "The cafeteria started with 23 apples. They used 20, so they had 23 - 20 = 3 apples left. Then they bought 6 more, so 3 + 6 = 9.",
                "answer": "9"
            }
        ]
    
    def format_prompt(self, question):
        prompt = "Solve these step by step:\n\n"
        
        # Add examples
        for example in self.examples:
            prompt += f"Q: {example['question']}\n"
            prompt += f"A: Let's think step by step. {example['reasoning']} The answer is {example['answer']}.\n\n"
        
        # Add new question
        prompt += f"Q: {question}\n"
        prompt += "A: Let's think step by step."
        
        return prompt

def extract_answer(response):
    """Extract final answer from chain-of-thought response"""
    # Look for patterns like "The answer is X" or "Therefore, X"
    import re
    
    patterns = [
        r"The answer is (\d+)",
        r"Therefore,? (\d+)",
        r"So the answer is (\d+)",
        r"= (\d+)$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    return None
```

### 2. **Code Generation**
```python
class CodeGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_code(self, prompt, language="python", max_length=200):
        """Generate code from natural language description"""
        
        # Format prompt for code generation
        formatted_prompt = f"""
# Language: {language}
# Task: {prompt}
# Code:

```{language}
"""
        
        # Generate code
        generated = self.model.generate(
            formatted_prompt,
            max_length=max_length,
            temperature=0.2,  # Lower temperature for more deterministic code
            top_p=0.9,
            stop_sequences=[f"```", "\n\n#"]
        )
        
        # Extract code block
        code = generated.split("```")[1] if "```" in generated else generated
        return code.strip()
    
    def explain_code(self, code):
        """Generate explanation for given code"""
        prompt = f"""
Explain what this code does:

```python
{code}
```

Explanation:
"""
        
        explanation = self.model.generate(
            prompt,
            max_length=150,
            temperature=0.7
        )
        
        return explanation
    
    def debug_code(self, code, error_message):
        """Help debug code given error message"""
        prompt = f"""
The following code has an error:

```python
{code}
```

Error message: {error_message}

Fixed code:

```python
"""
        
        fixed_code = self.model.generate(
            prompt,
            max_length=200,
            temperature=0.3
        )
        
        return fixed_code
```

### 3. **Retrieval-Augmented Generation (RAG)**
```python
class RAGSystem:
    def __init__(self, llm, retriever, max_context_length=2048):
        self.llm = llm
        self.retriever = retriever
        self.max_context_length = max_context_length
    
    def retrieve_and_generate(self, query, top_k=5):
        """Retrieve relevant documents and generate answer"""
        
        # Retrieve relevant documents
        documents = self.retriever.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.format_context(documents)
        
        # Create prompt
        prompt = f"""
Context:
{context}

Question: {query}

Answer based on the context above:
"""
        
        # Generate answer
        answer = self.llm.generate(
            prompt,
            max_length=200,
            temperature=0.7
        )
        
        return {
            'answer': answer,
            'sources': documents,
            'context': context
        }
    
    def format_context(self, documents):
        """Format retrieved documents as context"""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"[{i+1}] {doc['title']}: {doc['content']}"
            
            # Check if adding this document exceeds context length
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)

class SimpleRetriever:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        self.embeddings = self.compute_embeddings()
    
    def compute_embeddings(self):
        """Compute embeddings for all documents"""
        embeddings = []
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']}"
            embedding = self.embedding_model.encode(text)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def retrieve(self, query, top_k=5):
        """Retrieve top-k most similar documents"""
        query_embedding = self.embedding_model.encode(query)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top documents
        return [self.documents[i] for i in top_indices]
```

## 📊 Evaluation and Benchmarking

### 1. **Comprehensive Evaluation Suite**
```python
class LLMEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_perplexity(self, test_data):
        """Compute perplexity on test data"""
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_data:
                input_ids = batch['input_ids']
                
                # Compute loss
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def evaluate_few_shot(self, task_data, num_shots=5):
        """Evaluate few-shot learning performance"""
        results = []
        
        for task in task_data:
            # Create few-shot prompt
            prompt = self.create_few_shot_prompt(task['examples'][:num_shots])
            prompt += f"\nInput: {task['test_input']}\nOutput:"
            
            # Generate prediction
            prediction = self.model.generate(prompt, max_length=50, temperature=0.1)
            
            # Evaluate
            correct = self.evaluate_prediction(prediction, task['test_output'], task['task_type'])
            results.append(correct)
        
        accuracy = sum(results) / len(results)
        return accuracy
    
    def create_few_shot_prompt(self, examples):
        """Create few-shot prompt from examples"""
        prompt = "Here are some examples:\n\n"
        
        for i, example in enumerate(examples):
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        return prompt
    
    def evaluate_prediction(self, prediction, target, task_type):
        """Evaluate prediction based on task type"""
        if task_type == 'classification':
            return prediction.strip().lower() == target.strip().lower()
        elif task_type == 'generation':
            # Use BLEU or similar metric
            return self.compute_bleu(prediction, target) > 0.5
        else:
            return prediction.strip() == target.strip()
    
    def compute_bleu(self, prediction, reference):
        """Compute BLEU score"""
        # Simplified BLEU implementation
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # 1-gram precision
        matches = sum(1 for token in pred_tokens if token in ref_tokens)
        precision = matches / len(pred_tokens)
        
        return precision
```

### 2. **Safety and Bias Evaluation**
```python
class SafetyEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Load safety datasets
        self.toxic_prompts = self.load_toxic_prompts()
        self.bias_prompts = self.load_bias_prompts()
    
    def evaluate_toxicity(self, num_samples=100):
        """Evaluate model's tendency to generate toxic content"""
        toxic_responses = 0
        
        for prompt in self.toxic_prompts[:num_samples]:
            response = self.model.generate(prompt, max_length=100, temperature=0.7)
            
            if self.is_toxic(response):
                toxic_responses += 1
        
        toxicity_rate = toxic_responses / num_samples
        return toxicity_rate
    
    def evaluate_bias(self, demographic_groups):
        """Evaluate bias across demographic groups"""
        bias_scores = {}
        
        for group in demographic_groups:
            group_prompts = [p.format(group=group) for p in self.bias_prompts]
            
            sentiment_scores = []
            for prompt in group_prompts:
                response = self.model.generate(prompt, max_length=50, temperature=0.7)
                sentiment = self.compute_sentiment(response)
                sentiment_scores.append(sentiment)
            
            bias_scores[group] = np.mean(sentiment_scores)
        
        # Compute bias as variance across groups
        bias_variance = np.var(list(bias_scores.values()))
        return bias_scores, bias_variance
    
    def is_toxic(self, text):
        """Check if text is toxic (simplified)"""
        toxic_keywords = ['hate', 'violence', 'harmful']  # Simplified
        return any(keyword in text.lower() for keyword in toxic_keywords)
    
    def compute_sentiment(self, text):
        """Compute sentiment score (simplified)"""
        positive_words = ['good', 'great', 'excellent', 'positive']
        negative_words = ['bad', 'terrible', 'awful', 'negative']
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        return (pos_count - neg_count) / max(1, pos_count + neg_count)
```

## 🎓 Learning Path

### Week 1: LLM Fundamentals
- Understand transformer architecture for LLMs
- Learn about scaling laws and emergent abilities
- Implement basic GPT-style model

### Week 2: Training and Fine-tuning
- Master pre-training strategies
- Implement parameter-efficient fine-tuning
- Learn instruction tuning techniques

### Week 3: Advanced Techniques
- Understand RLHF and alignment
- Master prompt engineering
- Implement chain-of-thought reasoning

### Week 4: Applications and Deployment
- Build LLM applications
- Learn deployment optimization
- Understand safety and evaluation

## 📚 Additional Resources

### Foundational Papers
- "Language Models are Few-Shot Learners" (GPT-3)
- "Training language models to follow instructions with human feedback" (InstructGPT)
- "Constitutional AI: Harmlessness from AI Feedback" (Claude)
- "PaLM: Scaling Language Modeling with Pathways"

### Training and Scaling
- "Training Compute-Optimal Large Language Models" (Chinchilla)
- "Scaling Laws for Neural Language Models" (OpenAI)
- "An Empirical Analysis of Training Protocols for Neural Language Models"

### Fine-tuning and Alignment
- "LoRA: Low-Rank Adaptation of Large Language Models"
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Self-Instruct: Aligning Language Model with Self Generated Instructions"

### Practical Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

**Next Module**: [Diffusion Models](../06-Diffusion-Models/) →

*Ready to master the models that are reshaping AI? Let's build and fine-tune large language models! 🤖📚*