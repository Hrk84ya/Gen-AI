"""
Intent Classification Module

This module implements intent classification using various approaches:
- Traditional ML with TF-IDF features
- BERT-based transformer models
- Ensemble methods combining multiple classifiers
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """Text preprocessing utilities for intent classification"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                  lemmatize: bool = True) -> str:
        """Complete text preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract additional features from text"""
        features = {}
        
        # Length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Question indicators
        features['has_question_mark'] = '?' in text
        features['starts_with_wh'] = text.lower().startswith(('what', 'when', 'where', 'who', 'why', 'how'))
        
        # Sentiment indicators (simple)
        positive_words = ['good', 'great', 'excellent', 'love', 'like', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'worst']
        
        features['positive_words'] = sum(1 for word in positive_words if word in text.lower())
        features['negative_words'] = sum(1 for word in negative_words if word in text.lower())
        
        # Urgency indicators
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'help', 'problem']
        features['urgency_score'] = sum(1 for word in urgent_words if word in text.lower())
        
        return features


class TraditionalIntentClassifier:
    """Traditional ML-based intent classifier using TF-IDF and sklearn"""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        self.model_type = model_type
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize model based on type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.is_trained = False
    
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data"""
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Create label encoding
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        # Encode labels
        encoded_labels = [self.label_encoder[label] for label in labels]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(encoded_labels)
        
        return X, y
    
    def train(self, texts: List[str], labels: List[str], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the intent classifier"""
        print("Preparing training data...")
        X, y = self.prepare_data(texts, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_accuracy = (val_predictions == y_val).mean()
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        self.is_trained = True
        
        return {
            'validation_accuracy': val_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'num_classes': len(self.label_encoder),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single text"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities.max()
        
        # Decode label
        intent = self.reverse_label_encoder[prediction]
        
        return intent, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict intents for multiple texts"""
        results = []
        for text in texts:
            intent, confidence = self.predict(text)
            results.append((intent, confidence))
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Get feature importance for each class"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'coef_'):
            # For linear models like LogisticRegression
            importance_dict = {}
            
            for class_idx, class_name in self.reverse_label_encoder.items():
                if len(self.model.coef_.shape) == 1:  # Binary classification
                    coefficients = self.model.coef_
                else:  # Multi-class
                    coefficients = self.model.coef_[class_idx]
                
                # Get top positive and negative features
                feature_importance = list(zip(feature_names, coefficients))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                importance_dict[class_name] = feature_importance[:top_n]
            
            return importance_dict
        
        elif hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            feature_importance = list(zip(feature_names, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {'overall': feature_importance[:top_n]}
        
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.reverse_label_encoder = model_data['reverse_label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']


class BERTIntentClassifier:
    """BERT-based intent classifier using transformers"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Prepare data for BERT training"""
        # Create label encoding
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        # Encode labels
        encoded_labels = [self.label_encoder[label] for label in labels]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(encoded_labels)
        }
    
    def train(self, texts: List[str], labels: List[str], 
              validation_split: float = 0.2, epochs: int = 3) -> Dict[str, Any]:
        """Train BERT model"""
        print("Preparing BERT training data...")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42
        )
        
        # Prepare datasets
        train_data = self.prepare_data(train_texts, train_labels)
        val_data = self.prepare_data(val_texts, val_labels)
        
        # Initialize model
        num_labels = len(self.label_encoder)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_data)
        val_dataset = IntentDataset(val_data)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        print("Training BERT model...")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        self.is_trained = True
        
        return {
            'eval_loss': eval_results['eval_loss'],
            'num_classes': num_labels,
            'training_samples': len(train_texts),
            'validation_samples': len(val_texts)
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single text"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(predictions, dim=-1)
        
        # Decode label
        intent = self.reverse_label_encoder[predicted_class.item()]
        
        return intent, confidence.item()
    
    def save_model(self, filepath: str):
        """Save BERT model"""
        if self.model and self.tokenizer:
            self.model.save_pretrained(filepath)
            self.tokenizer.save_pretrained(filepath)
            
            # Save label encoders
            with open(f"{filepath}/label_encoders.json", 'w') as f:
                json.dump({
                    'label_encoder': self.label_encoder,
                    'reverse_label_encoder': self.reverse_label_encoder
                }, f)
    
    def load_model(self, filepath: str):
        """Load BERT model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        
        # Load label encoders
        with open(f"{filepath}/label_encoders.json", 'r') as f:
            encoders = json.load(f)
            self.label_encoder = encoders['label_encoder']
            # Convert string keys back to integers for reverse encoder
            self.reverse_label_encoder = {int(k): v for k, v in encoders['reverse_label_encoder'].items()}
        
        self.is_trained = True


class IntentDataset(torch.utils.data.Dataset):
    """Dataset class for BERT training"""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'labels': self.data['labels'][idx]
        }
    
    def __len__(self):
        return len(self.data['labels'])


class EnsembleIntentClassifier:
    """Ensemble classifier combining multiple models"""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.is_trained = all(getattr(model, 'is_trained', False) for model in models)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict using ensemble of models"""
        if not self.is_trained:
            raise ValueError("Not all models are trained.")
        
        predictions = []
        confidences = []
        
        for model in self.models:
            intent, confidence = model.predict(text)
            predictions.append(intent)
            confidences.append(confidence)
        
        # Weighted voting
        intent_scores = {}
        for i, (intent, confidence) in enumerate(zip(predictions, confidences)):
            weighted_confidence = confidence * self.weights[i]
            if intent in intent_scores:
                intent_scores[intent] += weighted_confidence
            else:
                intent_scores[intent] = weighted_confidence
        
        # Get best prediction
        best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        best_confidence = intent_scores[best_intent] / sum(self.weights)
        
        return best_intent, best_confidence


class IntentClassifier:
    """Main intent classifier interface"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'traditional'):
        self.model_type = model_type
        self.model_path = model_path
        
        if model_type == 'traditional':
            self.classifier = TraditionalIntentClassifier()
        elif model_type == 'bert':
            self.classifier = BERTIntentClassifier()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def train(self, training_data: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Train the classifier"""
        texts = [item['text'] for item in training_data]
        labels = [item['intent'] for item in training_data]
        
        return self.classifier.train(texts, labels, **kwargs)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for text"""
        return self.classifier.predict(text)
    
    def save_model(self, filepath: str):
        """Save model"""
        self.classifier.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load model"""
        self.classifier.load_model(filepath)


# Example usage and testing
if __name__ == "__main__":
    # Sample training data
    training_data = [
        {"text": "Hello", "intent": "greeting"},
        {"text": "Hi there", "intent": "greeting"},
        {"text": "Good morning", "intent": "greeting"},
        {"text": "I want to book a flight", "intent": "book_flight"},
        {"text": "Can you help me book a ticket?", "intent": "book_flight"},
        {"text": "I need to make a reservation", "intent": "book_flight"},
        {"text": "What's the weather like?", "intent": "weather"},
        {"text": "Is it going to rain today?", "intent": "weather"},
        {"text": "Tell me about the weather", "intent": "weather"},
        {"text": "Thank you", "intent": "thanks"},
        {"text": "Thanks a lot", "intent": "thanks"},
        {"text": "I appreciate your help", "intent": "thanks"},
        {"text": "Goodbye", "intent": "goodbye"},
        {"text": "See you later", "intent": "goodbye"},
        {"text": "Bye", "intent": "goodbye"}
    ]
    
    print("=== Testing Intent Classification ===")
    
    # Test traditional classifier
    print("\n1. Testing Traditional Classifier")
    traditional_classifier = IntentClassifier(model_type='traditional')
    
    # Train
    results = traditional_classifier.train(training_data)
    print(f"Training results: {results}")
    
    # Test predictions
    test_texts = [
        "Hi",
        "I want to book a flight to Paris",
        "What's the weather forecast?",
        "Thank you very much"
    ]
    
    for text in test_texts:
        intent, confidence = traditional_classifier.predict(text)
        print(f"Text: '{text}' -> Intent: {intent} (confidence: {confidence:.3f})")
    
    # Save model
    traditional_classifier.save_model("models/intent_traditional.pkl")
    print("Model saved successfully")
    
    print("\n✅ Intent classification tests completed!")