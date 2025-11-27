# 🎯 Prompt Engineering

## 🎯 Learning Objectives
- Master advanced prompting techniques
- Understand prompt design principles
- Build effective AI applications with prompts
- Optimize model performance through prompting

## 🧠 Fundamentals of Prompt Engineering

### What is Prompt Engineering?
Prompt engineering is the practice of designing inputs to get desired outputs from language models.

```python
# Basic prompt structure
prompt = """
Task: [What you want the model to do]
Context: [Relevant background information]
Input: [The specific input to process]
Format: [How you want the output structured]
"""
```

## 🎨 Core Prompting Techniques

### 1. Zero-Shot Prompting
```python
prompt = """
Classify the sentiment of this text as positive, negative, or neutral:
Text: "I love this new restaurant!"
Sentiment:
"""
# Output: positive
```

### 2. Few-Shot Prompting
```python
prompt = """
Classify the sentiment of these texts:

Text: "This movie was amazing!"
Sentiment: positive

Text: "I hate waiting in long lines."
Sentiment: negative

Text: "The weather is okay today."
Sentiment: neutral

Text: "This book changed my life!"
Sentiment:
"""
# Output: positive
```

### 3. Chain-of-Thought (CoT)
```python
prompt = """
Question: A store has 15 apples. They sell 7 apples in the morning and 3 apples in the afternoon. How many apples are left?

Let me think step by step:
1. The store starts with 15 apples
2. They sell 7 apples in the morning: 15 - 7 = 8 apples left
3. They sell 3 apples in the afternoon: 8 - 3 = 5 apples left
4. Therefore, 5 apples are left.

Question: A library has 120 books. They lend out 45 books on Monday and 23 books on Tuesday. How many books are left?

Let me think step by step:
"""
```

### 4. Tree of Thoughts
```python
prompt = """
Problem: Plan a 3-day trip to Paris for a couple on a budget.

Let me explore different approaches:

Approach 1 - Focus on free attractions:
- Day 1: Walk along Seine, visit Notre-Dame area, Sacré-Cœur
- Day 2: Louvre (free first Sunday), Tuileries Garden, Champs-Élysées
- Day 3: Montmartre, street art tour, local markets

Approach 2 - Mix of paid and free:
- Day 1: Eiffel Tower (paid), Seine river walk
- Day 2: Louvre Museum (paid), Latin Quarter exploration
- Day 3: Versailles day trip (paid transport)

Approach 3 - Cultural immersion:
- Day 1: Local neighborhood exploration, café culture
- Day 2: Museums with student discounts, local food markets
- Day 3: Day trip to nearby towns by train

Best approach considering budget constraints: Approach 1 with selective elements from others.
"""
```

## 🔧 Advanced Techniques

### Role-Based Prompting
```python
system_prompt = """
You are a senior software engineer with 10 years of experience in Python and machine learning. 
You write clean, efficient code and always consider best practices, error handling, and documentation.
When asked to write code, you:
1. Write clean, readable code with proper naming
2. Include error handling where appropriate
3. Add docstrings and comments
4. Consider edge cases
5. Suggest improvements or alternatives
"""

user_prompt = """
Write a function to calculate the moving average of a list of numbers.
"""
```

### Instruction Following with Examples
```python
prompt = """
You are a helpful assistant that converts natural language to SQL queries.

Examples:
Natural Language: "Show me all customers from New York"
SQL: SELECT * FROM customers WHERE city = 'New York';

Natural Language: "Count how many orders were placed last month"
SQL: SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH);

Natural Language: "Find the top 5 products by sales"
SQL: SELECT product_name, SUM(quantity * price) as total_sales FROM order_items oi JOIN products p ON oi.product_id = p.id GROUP BY product_name ORDER BY total_sales DESC LIMIT 5;

Now convert this:
Natural Language: "Show me customers who have spent more than $1000"
SQL:
"""
```

### Prompt Chaining
```python
# Step 1: Extract key information
extraction_prompt = """
Extract the key information from this customer review:
Review: "The pizza was cold when it arrived, but the delivery guy was very friendly. The taste was okay, not great. Would order again if they fix the temperature issue."

Extract:
- Food quality:
- Service quality:
- Delivery experience:
- Overall sentiment:
"""

# Step 2: Generate response based on extraction
response_prompt = """
Based on this customer feedback analysis:
- Food quality: Cold pizza, okay taste
- Service quality: Friendly delivery person
- Delivery experience: Food arrived cold
- Overall sentiment: Mixed, willing to try again

Write a professional response addressing their concerns:
"""
```

## 🎯 Prompt Optimization Strategies

### 1. Iterative Refinement
```python
# Version 1 (Basic)
prompt_v1 = "Summarize this article."

# Version 2 (More specific)
prompt_v2 = "Summarize this article in 3 bullet points focusing on the main findings."

# Version 3 (With format)
prompt_v3 = """
Summarize this article in exactly 3 bullet points. Focus on:
1. Main research finding
2. Methodology used
3. Practical implications

Format each bullet point as: "• [Summary]"
"""

# Version 4 (With examples)
prompt_v4 = """
Summarize this article in exactly 3 bullet points following this format:

Example:
• Main finding: Study shows 40% improvement in learning with spaced repetition
• Methodology: Randomized controlled trial with 200 students over 6 months
• Implications: Educational institutions should implement spaced learning schedules

Now summarize this article:
[Article text]

Summary:
"""
```

### 2. Temperature and Parameter Tuning
```python
import openai

# For creative tasks - higher temperature
creative_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a creative story about AI"}],
    temperature=0.9,  # More creative, less predictable
    max_tokens=500
)

# For factual tasks - lower temperature
factual_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.1,  # More focused, predictable
    max_tokens=50
)
```

## 🛠️ Practical Applications

### 1. Content Generation
```python
content_prompt = """
Write a blog post about sustainable living. Include:

Structure:
- Engaging headline
- Introduction (hook + thesis)
- 3 main points with examples
- Conclusion with call-to-action

Tone: Conversational but informative
Length: ~500 words
Target audience: Young professionals interested in environmental issues

Topic focus: Simple changes that make a big impact
"""
```

### 2. Code Generation and Review
```python
code_prompt = """
Create a Python class for a simple task manager with these requirements:

Functionality:
- Add tasks with priority levels (high, medium, low)
- Mark tasks as complete
- List tasks by priority
- Get overdue tasks

Requirements:
- Use proper OOP principles
- Include error handling
- Add type hints
- Write docstrings
- Include example usage

Class name: TaskManager
"""
```

### 3. Data Analysis
```python
analysis_prompt = """
Analyze this sales data and provide insights:

Data: [CSV data or description]

Please provide:
1. Key trends and patterns
2. Top performing products/categories
3. Seasonal variations
4. Recommendations for improvement
5. Potential risks or concerns

Format your response with clear headings and bullet points.
Include specific numbers and percentages where relevant.
"""
```

## 🔍 Prompt Evaluation and Testing

### A/B Testing Prompts
```python
def test_prompts(prompts, test_cases, model="gpt-3.5-turbo"):
    results = {}
    
    for i, prompt in enumerate(prompts):
        results[f"prompt_{i+1}"] = []
        
        for test_case in test_cases:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt.format(**test_case)}],
                temperature=0.1
            )
            results[f"prompt_{i+1}"].append(response.choices[0].message.content)
    
    return results

# Example usage
prompts = [
    "Classify this email as spam or not spam: {email}",
    "Determine if this email is spam (unwanted/promotional) or legitimate: {email}",
    "Is this email spam? Answer with 'spam' or 'not spam': {email}"
]

test_cases = [
    {"email": "Congratulations! You've won $1,000,000! Click here now!"},
    {"email": "Hi John, can we reschedule our meeting to 3 PM tomorrow?"}
]

results = test_prompts(prompts, test_cases)
```

### Evaluation Metrics
```python
def evaluate_prompt_performance(responses, expected_outputs):
    metrics = {
        'accuracy': 0,
        'consistency': 0,
        'relevance': 0
    }
    
    # Calculate accuracy
    correct = sum(1 for r, e in zip(responses, expected_outputs) if r.strip().lower() == e.strip().lower())
    metrics['accuracy'] = correct / len(responses)
    
    # Calculate consistency (how similar responses are for similar inputs)
    # Implementation depends on specific use case
    
    return metrics
```

## 🚀 Advanced Prompt Patterns

### 1. Constitutional AI Pattern
```python
constitutional_prompt = """
I want you to respond to user queries, but first check if your response follows these principles:
1. Be helpful and informative
2. Be honest about limitations
3. Avoid harmful or biased content
4. Respect privacy and confidentiality
5. Encourage critical thinking

User query: {query}

First, analyze if a direct response would violate any principles above.
Then provide your response, noting any limitations or caveats.
"""
```

### 2. Metacognitive Prompting
```python
metacognitive_prompt = """
Before answering this question, I want you to:
1. Identify what type of question this is
2. Consider what knowledge or reasoning is needed
3. Think about potential pitfalls or biases
4. Plan your approach

Question: {question}

Analysis:
Type of question:
Knowledge needed:
Potential issues:
Approach:

Answer:
"""
```

### 3. Socratic Method
```python
socratic_prompt = """
Instead of giving me the answer directly, guide me to discover it myself using the Socratic method.

Ask me probing questions that help me think through the problem step by step.
Don't give away the answer, but help me reason through it.

My question: How do neural networks learn?

Your first guiding question:
"""
```

## 📊 Prompt Engineering Tools

### 1. Prompt Templates
```python
class PromptTemplate:
    def __init__(self, template, required_vars):
        self.template = template
        self.required_vars = required_vars
    
    def format(self, **kwargs):
        missing_vars = set(self.required_vars) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return self.template.format(**kwargs)

# Usage
classification_template = PromptTemplate(
    template="""
    Classify the following {item_type} into one of these categories: {categories}
    
    {item_type}: {item}
    
    Category:
    """,
    required_vars=['item_type', 'categories', 'item']
)

prompt = classification_template.format(
    item_type="email",
    categories="spam, not spam",
    item="Buy now! Limited time offer!"
)
```

### 2. Prompt Optimization
```python
def optimize_prompt(base_prompt, test_cases, target_metric='accuracy'):
    variations = [
        base_prompt,
        f"Please {base_prompt.lower()}",
        f"{base_prompt}\n\nThink step by step:",
        f"You are an expert. {base_prompt}",
        f"{base_prompt}\n\nBe precise and accurate:"
    ]
    
    best_prompt = base_prompt
    best_score = 0
    
    for prompt in variations:
        score = evaluate_prompt(prompt, test_cases, target_metric)
        if score > best_score:
            best_score = score
            best_prompt = prompt
    
    return best_prompt, best_score
```

## 🎯 Best Practices

### Do's
- ✅ Be specific and clear in instructions
- ✅ Provide examples when possible
- ✅ Use consistent formatting
- ✅ Test with diverse inputs
- ✅ Iterate and refine prompts
- ✅ Consider the model's training data cutoff

### Don'ts
- ❌ Make prompts unnecessarily long
- ❌ Use ambiguous language
- ❌ Assume the model knows context
- ❌ Ignore edge cases
- ❌ Use biased or leading questions

## 📚 Resources

### Tools
- [OpenAI Playground](https://platform.openai.com/playground)
- [PromptBase](https://promptbase.com/)
- [LangChain](https://python.langchain.com/)
- [Guidance](https://github.com/microsoft/guidance)

### Research Papers
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- "Constitutional AI: Harmlessness from AI Feedback"

---
*Continue to [AI Agents & RAG](../04-AI-Agents/) →*