"""
Advanced Prompt Engineering Templates and Utilities

This module provides a comprehensive framework for creating, managing,
and optimizing prompts for large language models.
"""

import json
import re
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from abc import ABC, abstractmethod


class PromptType(Enum):
    """Types of prompts for different use cases"""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    CODE = "code"
    ANALYSIS = "analysis"


@dataclass
class PromptMetadata:
    """Metadata for prompt templates"""
    name: str
    description: str
    prompt_type: PromptType
    version: str = "1.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    model_compatibility: List[str] = field(default_factory=list)
    expected_output_format: str = ""
    performance_notes: str = ""


class PromptTemplate:
    """
    Advanced prompt template with validation, formatting, and optimization features
    """
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        required_variables: Optional[List[str]] = None,
        optional_variables: Optional[List[str]] = None,
        validation_rules: Optional[Dict[str, Callable]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ):
        self.template = template
        self.metadata = metadata
        self.required_variables = required_variables or []
        self.optional_variables = optional_variables or []
        self.validation_rules = validation_rules or {}
        self.examples = examples or []
        
        # Extract variables from template
        self._extract_variables()
        
        # Generate template hash for caching
        self.template_hash = self._generate_hash()
    
    def _extract_variables(self):
        """Extract variable names from template using regex"""
        pattern = r'\{(\w+)\}'
        found_variables = set(re.findall(pattern, self.template))
        
        # Auto-detect required variables if not specified
        if not self.required_variables:
            self.required_variables = list(found_variables)
    
    def _generate_hash(self) -> str:
        """Generate unique hash for template"""
        content = f"{self.template}{self.metadata.name}{self.metadata.version}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def validate_variables(self, variables: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate input variables against template requirements
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        # Check required variables
        missing_required = set(self.required_variables) - set(variables.keys())
        if missing_required:
            errors.append(f"Missing required variables: {missing_required}")
        
        # Check for extra variables
        all_expected = set(self.required_variables + self.optional_variables)
        extra_variables = set(variables.keys()) - all_expected
        if extra_variables:
            warnings.append(f"Unexpected variables provided: {extra_variables}")
        
        # Apply custom validation rules
        for var_name, rule in self.validation_rules.items():
            if var_name in variables:
                try:
                    if not rule(variables[var_name]):
                        errors.append(f"Validation failed for variable '{var_name}'")
                except Exception as e:
                    errors.append(f"Validation error for '{var_name}': {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def format(self, **variables) -> str:
        """
        Format template with provided variables
        
        Args:
            **variables: Variable values to substitute in template
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If validation fails
        """
        validation_result = self.validate_variables(variables)
        
        if validation_result["errors"]:
            raise ValueError(f"Template validation failed: {validation_result['errors']}")
        
        # Print warnings if any
        if validation_result["warnings"]:
            print(f"Warnings: {validation_result['warnings']}")
        
        try:
            return self.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e}")
    
    def get_example_prompt(self, example_index: int = 0) -> str:
        """Get formatted prompt using example data"""
        if not self.examples or example_index >= len(self.examples):
            raise ValueError(f"No example available at index {example_index}")
        
        return self.format(**self.examples[example_index])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization"""
        return {
            "template": self.template,
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "prompt_type": self.metadata.prompt_type.value,
                "version": self.metadata.version,
                "author": self.metadata.author,
                "tags": self.metadata.tags,
                "created_at": self.metadata.created_at,
                "model_compatibility": self.metadata.model_compatibility,
                "expected_output_format": self.metadata.expected_output_format,
                "performance_notes": self.metadata.performance_notes
            },
            "required_variables": self.required_variables,
            "optional_variables": self.optional_variables,
            "examples": self.examples,
            "template_hash": self.template_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary"""
        metadata = PromptMetadata(
            name=data["metadata"]["name"],
            description=data["metadata"]["description"],
            prompt_type=PromptType(data["metadata"]["prompt_type"]),
            version=data["metadata"].get("version", "1.0"),
            author=data["metadata"].get("author", ""),
            tags=data["metadata"].get("tags", []),
            created_at=data["metadata"].get("created_at", ""),
            model_compatibility=data["metadata"].get("model_compatibility", []),
            expected_output_format=data["metadata"].get("expected_output_format", ""),
            performance_notes=data["metadata"].get("performance_notes", "")
        )
        
        return cls(
            template=data["template"],
            metadata=metadata,
            required_variables=data.get("required_variables", []),
            optional_variables=data.get("optional_variables", []),
            examples=data.get("examples", [])
        )


class ChainOfThoughtTemplate(PromptTemplate):
    """Specialized template for Chain-of-Thought prompting"""
    
    def __init__(self, problem_template: str, reasoning_examples: List[Dict[str, str]], **kwargs):
        self.reasoning_examples = reasoning_examples
        
        # Build CoT template
        cot_template = self._build_cot_template(problem_template, reasoning_examples)
        
        super().__init__(cot_template, **kwargs)
    
    def _build_cot_template(self, problem_template: str, examples: List[Dict[str, str]]) -> str:
        """Build Chain-of-Thought template with examples"""
        template_parts = []
        
        # Add examples
        for example in examples:
            template_parts.append(f"Problem: {example['problem']}")
            template_parts.append(f"Let me think step by step:")
            template_parts.append(example['reasoning'])
            template_parts.append(f"Answer: {example['answer']}")
            template_parts.append("")
        
        # Add the actual problem
        template_parts.append(f"Problem: {problem_template}")
        template_parts.append("Let me think step by step:")
        
        return "\n".join(template_parts)


class FewShotTemplate(PromptTemplate):
    """Template for few-shot learning with examples"""
    
    def __init__(self, task_description: str, examples: List[Dict[str, str]], 
                 input_template: str, **kwargs):
        self.task_description = task_description
        self.shot_examples = examples
        
        # Build few-shot template
        few_shot_template = self._build_few_shot_template(
            task_description, examples, input_template
        )
        
        super().__init__(few_shot_template, **kwargs)
    
    def _build_few_shot_template(self, description: str, examples: List[Dict[str, str]], 
                               input_template: str) -> str:
        """Build few-shot template with examples"""
        template_parts = [description, ""]
        
        # Add examples
        for example in examples:
            template_parts.append(f"Input: {example['input']}")
            template_parts.append(f"Output: {example['output']}")
            template_parts.append("")
        
        # Add the actual input
        template_parts.append(f"Input: {input_template}")
        template_parts.append("Output:")
        
        return "\n".join(template_parts)


class PromptLibrary:
    """Library for managing and organizing prompt templates"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.categories: Dict[PromptType, List[str]] = {}
    
    def add_template(self, template: PromptTemplate):
        """Add template to library"""
        self.templates[template.metadata.name] = template
        
        # Update categories
        prompt_type = template.metadata.prompt_type
        if prompt_type not in self.categories:
            self.categories[prompt_type] = []
        
        if template.metadata.name not in self.categories[prompt_type]:
            self.categories[prompt_type].append(template.metadata.name)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def list_templates(self, prompt_type: Optional[PromptType] = None, 
                      tags: Optional[List[str]] = None) -> List[str]:
        """List templates by type and/or tags"""
        templates = list(self.templates.keys())
        
        if prompt_type:
            templates = [name for name in templates 
                        if self.templates[name].metadata.prompt_type == prompt_type]
        
        if tags:
            templates = [name for name in templates 
                        if any(tag in self.templates[name].metadata.tags for tag in tags)]
        
        return templates
    
    def search_templates(self, query: str) -> List[str]:
        """Search templates by name, description, or tags"""
        query = query.lower()
        results = []
        
        for name, template in self.templates.items():
            if (query in name.lower() or 
                query in template.metadata.description.lower() or
                any(query in tag.lower() for tag in template.metadata.tags)):
                results.append(name)
        
        return results
    
    def save_library(self, filepath: str):
        """Save library to JSON file"""
        library_data = {
            "templates": {name: template.to_dict() 
                         for name, template in self.templates.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(library_data, f, indent=2)
    
    def load_library(self, filepath: str):
        """Load library from JSON file"""
        with open(filepath, 'r') as f:
            library_data = json.load(f)
        
        for name, template_data in library_data["templates"].items():
            template = PromptTemplate.from_dict(template_data)
            self.add_template(template)


class PromptOptimizer:
    """Tools for optimizing prompt performance"""
    
    def __init__(self):
        self.optimization_history = []
    
    def generate_variations(self, base_prompt: str, variation_types: List[str]) -> List[str]:
        """
        Generate prompt variations for testing
        
        Args:
            base_prompt: Original prompt
            variation_types: Types of variations to generate
        
        Returns:
            List of prompt variations
        """
        variations = [base_prompt]  # Include original
        
        for variation_type in variation_types:
            if variation_type == "polite":
                variations.append(f"Please {base_prompt.lower()}")
            
            elif variation_type == "step_by_step":
                variations.append(f"{base_prompt}\n\nThink step by step:")
            
            elif variation_type == "expert":
                variations.append(f"You are an expert in this field. {base_prompt}")
            
            elif variation_type == "precise":
                variations.append(f"{base_prompt}\n\nBe precise and accurate:")
            
            elif variation_type == "creative":
                variations.append(f"{base_prompt}\n\nBe creative and think outside the box:")
            
            elif variation_type == "format_request":
                variations.append(f"{base_prompt}\n\nFormat your response clearly with headings and bullet points:")
            
            elif variation_type == "examples_request":
                variations.append(f"{base_prompt}\n\nProvide specific examples to illustrate your points:")
        
        return variations
    
    def a_b_test_prompts(self, prompts: List[str], test_cases: List[Dict[str, Any]], 
                        evaluation_function: Callable) -> Dict[str, Any]:
        """
        A/B test different prompt variations
        
        Args:
            prompts: List of prompt variations to test
            test_cases: Test cases to evaluate on
            evaluation_function: Function to evaluate prompt performance
        
        Returns:
            Results of A/B test
        """
        results = {}
        
        for i, prompt in enumerate(prompts):
            prompt_id = f"prompt_{i+1}"
            results[prompt_id] = {
                "prompt": prompt,
                "scores": [],
                "average_score": 0,
                "test_results": []
            }
            
            for test_case in test_cases:
                # Format prompt with test case
                formatted_prompt = prompt.format(**test_case.get("variables", {}))
                
                # Evaluate (this would typically call an LLM API)
                score = evaluation_function(formatted_prompt, test_case)
                
                results[prompt_id]["scores"].append(score)
                results[prompt_id]["test_results"].append({
                    "test_case": test_case,
                    "score": score
                })
            
            # Calculate average score
            if results[prompt_id]["scores"]:
                results[prompt_id]["average_score"] = sum(results[prompt_id]["scores"]) / len(results[prompt_id]["scores"])
        
        # Find best performing prompt
        best_prompt = max(results.keys(), key=lambda k: results[k]["average_score"])
        results["best_prompt"] = best_prompt
        results["best_score"] = results[best_prompt]["average_score"]
        
        return results


# Pre-built template library
def create_standard_library() -> PromptLibrary:
    """Create a library with standard prompt templates"""
    library = PromptLibrary()
    
    # Classification template
    classification_template = PromptTemplate(
        template="""
        Classify the following {item_type} into one of these categories: {categories}
        
        {item_type}: {item}
        
        Think about the key characteristics that define each category.
        
        Category:
        """,
        metadata=PromptMetadata(
            name="basic_classification",
            description="General purpose classification template",
            prompt_type=PromptType.CLASSIFICATION,
            tags=["classification", "general"],
            expected_output_format="Single category name"
        ),
        required_variables=["item_type", "categories", "item"],
        examples=[
            {
                "item_type": "email",
                "categories": "spam, not spam",
                "item": "Congratulations! You've won $1,000,000!"
            }
        ]
    )
    
    # Code generation template
    code_template = PromptTemplate(
        template="""
        Write a {language} function that {description}.
        
        Requirements:
        {requirements}
        
        Please include:
        - Proper error handling
        - Clear variable names
        - Comments explaining the logic
        - Example usage
        
        Function:
        """,
        metadata=PromptMetadata(
            name="code_generation",
            description="Template for generating code functions",
            prompt_type=PromptType.CODE,
            tags=["code", "programming", "function"],
            expected_output_format="Complete function with documentation"
        ),
        required_variables=["language", "description", "requirements"],
        examples=[
            {
                "language": "Python",
                "description": "calculates the factorial of a number",
                "requirements": "- Handle negative numbers gracefully\n- Use recursion\n- Include type hints"
            }
        ]
    )
    
    # Analysis template
    analysis_template = PromptTemplate(
        template="""
        Analyze the following {data_type} and provide insights:
        
        Data: {data}
        
        Please provide:
        1. Key patterns and trends
        2. Notable observations
        3. Potential implications
        4. Recommendations for action
        
        Structure your analysis with clear headings and support your insights with specific evidence from the data.
        
        Analysis:
        """,
        metadata=PromptMetadata(
            name="data_analysis",
            description="Template for analyzing data and providing insights",
            prompt_type=PromptType.ANALYSIS,
            tags=["analysis", "data", "insights"],
            expected_output_format="Structured analysis with headings and bullet points"
        ),
        required_variables=["data_type", "data"],
        examples=[
            {
                "data_type": "sales report",
                "data": "Q1 sales: $100k, Q2 sales: $150k, Q3 sales: $120k, Q4 sales: $180k"
            }
        ]
    )
    
    # Add templates to library
    library.add_template(classification_template)
    library.add_template(code_template)
    library.add_template(analysis_template)
    
    return library


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Prompt Engineering Framework ===")
    
    # Create standard library
    library = create_standard_library()
    
    # Test classification template
    print("\n1. Testing Classification Template")
    classification_template = library.get_template("basic_classification")
    
    prompt = classification_template.format(
        item_type="movie review",
        categories="positive, negative, neutral",
        item="This movie was absolutely fantastic! Great acting and storyline."
    )
    print("Generated prompt:")
    print(prompt)
    
    # Test code generation template
    print("\n2. Testing Code Generation Template")
    code_template = library.get_template("code_generation")
    
    code_prompt = code_template.format(
        language="Python",
        description="sorts a list of dictionaries by a specified key",
        requirements="- Handle empty lists\n- Allow reverse sorting\n- Support nested keys"
    )
    print("Generated code prompt:")
    print(code_prompt)
    
    # Test Chain-of-Thought template
    print("\n3. Testing Chain-of-Thought Template")
    
    cot_metadata = PromptMetadata(
        name="math_cot",
        description="Chain-of-thought for math problems",
        prompt_type=PromptType.REASONING
    )
    
    reasoning_examples = [
        {
            "problem": "What is 15% of 80?",
            "reasoning": "To find 15% of 80:\n1. Convert 15% to decimal: 15% = 0.15\n2. Multiply: 0.15 × 80 = 12",
            "answer": "12"
        }
    ]
    
    cot_template = ChainOfThoughtTemplate(
        problem_template="{problem}",
        reasoning_examples=reasoning_examples,
        metadata=cot_metadata,
        required_variables=["problem"]
    )
    
    math_prompt = cot_template.format(problem="What is 25% of 120?")
    print("Generated CoT prompt:")
    print(math_prompt)
    
    # Test prompt optimization
    print("\n4. Testing Prompt Optimization")
    optimizer = PromptOptimizer()
    
    base_prompt = "Summarize this article: {article}"
    variations = optimizer.generate_variations(
        base_prompt, 
        ["polite", "step_by_step", "expert", "precise"]
    )
    
    print("Generated variations:")
    for i, variation in enumerate(variations):
        print(f"{i+1}. {variation}")
    
    # Test library operations
    print("\n5. Testing Library Operations")
    print(f"Total templates: {len(library.templates)}")
    print(f"Classification templates: {library.list_templates(PromptType.CLASSIFICATION)}")
    print(f"Code templates: {library.list_templates(PromptType.CODE)}")
    
    # Search functionality
    search_results = library.search_templates("code")
    print(f"Search results for 'code': {search_results}")
    
    print("\n✅ All prompt engineering tests completed successfully!")