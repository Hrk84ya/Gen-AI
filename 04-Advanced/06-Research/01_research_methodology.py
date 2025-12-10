"""
AI Research Methodology Tools

This module provides tools and utilities for conducting rigorous AI research,
including experiment tracking, statistical analysis, reproducibility tools,
and evaluation frameworks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import hashlib
import pickle
import os
from datetime import datetime
import random
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class ExperimentTracker:
    """
    Comprehensive experiment tracking system for AI research.
    Tracks hyperparameters, metrics, artifacts, and ensures reproducibility.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        
        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize tracking data
        self.runs = []
        self.current_run = None
        
        # Load existing experiments if any
        self.load_experiments()
    
    def start_run(self, run_name: Optional[str] = None, config: Optional[Dict] = None) -> str:
        """Start a new experimental run"""
        if run_name is None:
            run_name = f"run_{len(self.runs) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_id = hashlib.md5(f"{self.experiment_name}_{run_name}".encode()).hexdigest()[:8]
        
        self.current_run = {
            'run_id': run_id,
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
            'config': config or {},
            'metrics': {},
            'artifacts': {},
            'logs': [],
            'status': 'running'
        }
        
        print(f"Started run: {run_name} (ID: {run_id})")
        return run_id
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        self.current_run['config'].update(config)
        print(f"Logged config: {list(config.keys())}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        if name not in self.current_run['metrics']:
            self.current_run['metrics'][name] = []
        
        metric_entry = {
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_run['metrics'][name].append(metric_entry)
    
    def log_artifact(self, name: str, artifact: Any, artifact_type: str = "pickle"):
        """Log an artifact (model, plot, data, etc.)"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        artifact_path = os.path.join(
            self.experiment_dir, 
            f"{self.current_run['run_id']}_{name}.{artifact_type}"
        )
        
        if artifact_type == "pickle":
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        elif artifact_type == "json":
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=2)
        elif artifact_type == "plot":
            artifact.savefig(artifact_path)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
        
        self.current_run['artifacts'][name] = {
            'path': artifact_path,
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Logged artifact: {name}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Log a message"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        log_entry = {
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_run['logs'].append(log_entry)
        print(f"[{level}] {message}")
    
    def end_run(self, status: str = "completed"):
        """End the current run"""
        if self.current_run is None:
            raise ValueError("No active run to end.")
        
        self.current_run['end_time'] = datetime.now().isoformat()
        self.current_run['status'] = status
        
        # Add to runs list
        self.runs.append(self.current_run.copy())
        
        # Save experiment data
        self.save_experiments()
        
        print(f"Ended run: {self.current_run['run_name']} with status: {status}")
        self.current_run = None
    
    def save_experiments(self):
        """Save experiment data to disk"""
        experiment_file = os.path.join(self.experiment_dir, "experiments.json")
        
        with open(experiment_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'runs': self.runs
            }, f, indent=2)
    
    def load_experiments(self):
        """Load existing experiment data"""
        experiment_file = os.path.join(self.experiment_dir, "experiments.json")
        
        if os.path.exists(experiment_file):
            with open(experiment_file, 'r') as f:
                data = json.load(f)
                self.runs = data.get('runs', [])
    
    def get_run_summary(self) -> pd.DataFrame:
        """Get summary of all runs as DataFrame"""
        if not self.runs:
            return pd.DataFrame()
        
        summary_data = []
        for run in self.runs:
            row = {
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'status': run['status'],
                'start_time': run['start_time']
            }
            
            # Add config parameters
            for key, value in run['config'].items():
                row[f'config_{key}'] = value
            
            # Add final metric values
            for metric_name, metric_values in run['metrics'].items():
                if metric_values:
                    row[f'final_{metric_name}'] = metric_values[-1]['value']
                    row[f'best_{metric_name}'] = max(m['value'] for m in metric_values)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def compare_runs(self, metric_name: str, run_ids: Optional[List[str]] = None):
        """Compare metric across runs"""
        if run_ids is None:
            runs_to_compare = self.runs
        else:
            runs_to_compare = [r for r in self.runs if r['run_id'] in run_ids]
        
        plt.figure(figsize=(12, 6))
        
        for run in runs_to_compare:
            if metric_name in run['metrics']:
                values = [m['value'] for m in run['metrics'][metric_name]]
                steps = [m['step'] or i for i, m in enumerate(run['metrics'][metric_name])]
                plt.plot(steps, values, label=f"{run['run_name']} ({run['run_id']})")
        
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'Comparison of {metric_name} across runs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class StatisticalAnalyzer:
    """
    Statistical analysis tools for AI research including significance testing,
    confidence intervals, and effect size calculations.
    """
    
    @staticmethod
    def compare_models(results1: List[float], results2: List[float], 
                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Compare two sets of model results using statistical tests.
        
        Args:
            results1: Results from model 1
            results2: Results from model 2
            alpha: Significance level
        
        Returns:
            Dictionary with statistical test results
        """
        results1, results2 = np.array(results1), np.array(results2)
        
        # Descriptive statistics
        stats_dict = {
            'model1_mean': np.mean(results1),
            'model1_std': np.std(results1),
            'model2_mean': np.mean(results2),
            'model2_std': np.std(results2),
            'difference': np.mean(results1) - np.mean(results2)
        }
        
        # Normality tests
        _, p_norm1 = stats.shapiro(results1)
        _, p_norm2 = stats.shapiro(results2)
        
        # Choose appropriate test
        if p_norm1 > alpha and p_norm2 > alpha:
            # Both normal - use t-test
            if len(results1) == len(results2):
                # Paired t-test
                statistic, p_value = stats.ttest_rel(results1, results2)
                test_used = "Paired t-test"
            else:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(results1, results2)
                test_used = "Independent t-test"
        else:
            # Non-normal - use Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
            test_used = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results1) - 1) * np.var(results1) + 
                             (len(results2) - 1) * np.var(results2)) / 
                            (len(results1) + len(results2) - 2))
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
        
        # Confidence interval for difference
        se_diff = np.sqrt(np.var(results1)/len(results1) + np.var(results2)/len(results2))
        ci_lower = stats_dict['difference'] - 1.96 * se_diff
        ci_upper = stats_dict['difference'] + 1.96 * se_diff
        
        stats_dict.update({
            'test_used': test_used,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size': StatisticalAnalyzer._interpret_cohens_d(cohens_d),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'normality_p1': p_norm1,
            'normality_p2': p_norm2
        })
        
        return stats_dict
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def multiple_comparison_correction(p_values: List[float], 
                                    method: str = "bonferroni") -> List[float]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        
        Returns:
            Corrected p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == "bonferroni":
            return np.minimum(p_values * n, 1.0)
        
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (n - i), 1.0)
            
            return corrected
        
        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * n / (i + 1), 1.0)
            
            return corrected
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, statistic: Callable,
                                    confidence: float = 0.95, 
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data
            statistic: Function to calculate statistic (e.g., np.mean)
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Lower and upper confidence bounds
        """
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper


class ReproducibilityManager:
    """
    Tools for ensuring reproducibility in AI research experiments.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.set_seeds()
        self.environment_info = self._get_environment_info()
    
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Collect environment information for reproducibility"""
        import platform
        import sys
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'seed': self.seed
        }
        
        # Try to get package versions
        try:
            import torch
            env_info['torch_version'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import numpy
            env_info['numpy_version'] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import sklearn
            env_info['sklearn_version'] = sklearn.__version__
        except ImportError:
            pass
        
        return env_info
    
    def save_environment(self, filepath: str):
        """Save environment information to file"""
        with open(filepath, 'w') as f:
            json.dump(self.environment_info, f, indent=2)
    
    def create_reproducible_split(self, data_size: int, 
                                train_ratio: float = 0.8,
                                val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create reproducible train/val/test splits"""
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices


class ModelEvaluator:
    """
    Comprehensive model evaluation framework with multiple metrics and visualizations.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            class_names: Names of classes (optional)
        
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (classification_report, confusion_matrix, 
                                   roc_auc_score, roc_curve, precision_recall_curve)
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1
        
        # Per-class metrics
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # ROC AUC (for binary or multiclass with probabilities)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    results['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                else:
                    # Multiclass
                    results['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None,
                            normalize: bool = False, title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        return plt.gcf()
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            scoring: Scoring metric
        
        Returns:
            Cross-validation results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'cv_folds': cv_folds,
            'scoring': scoring
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Research Methodology Tools ===")
    
    # Test Experiment Tracker
    print("\n1. Testing Experiment Tracker")
    tracker = ExperimentTracker("test_experiment")
    
    # Run a mock experiment
    run_id = tracker.start_run("test_run", {"learning_rate": 0.001, "batch_size": 32})
    
    # Log some metrics
    for epoch in range(5):
        accuracy = 0.5 + 0.1 * epoch + np.random.normal(0, 0.02)
        loss = 2.0 - 0.3 * epoch + np.random.normal(0, 0.1)
        
        tracker.log_metric("accuracy", accuracy, epoch)
        tracker.log_metric("loss", loss, epoch)
    
    tracker.log_message("Training completed successfully")
    tracker.end_run("completed")
    
    # Get summary
    summary = tracker.get_summary()
    print(f"Experiment summary shape: {summary.shape}")
    
    # Test Statistical Analyzer
    print("\n2. Testing Statistical Analyzer")
    
    # Generate mock results for two models
    model1_results = np.random.normal(0.85, 0.05, 20)
    model2_results = np.random.normal(0.82, 0.04, 20)
    
    comparison = StatisticalAnalyzer.compare_models(model1_results, model2_results)
    print(f"Model comparison p-value: {comparison['p_value']:.4f}")
    print(f"Effect size: {comparison['effect_size']}")
    print(f"Significant difference: {comparison['significant']}")
    
    # Test bootstrap CI
    ci_lower, ci_upper = StatisticalAnalyzer.bootstrap_confidence_interval(
        model1_results, np.mean, confidence=0.95
    )
    print(f"Bootstrap 95% CI for model 1 mean: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Test Reproducibility Manager
    print("\n3. Testing Reproducibility Manager")
    repro_manager = ReproducibilityManager(seed=42)
    
    # Generate reproducible data splits
    train_idx, val_idx, test_idx = repro_manager.create_reproducible_split(1000)
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Test Model Evaluator
    print("\n4. Testing Model Evaluator")
    evaluator = ModelEvaluator()
    
    # Generate mock classification data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = y_true.copy()
    # Add some noise
    noise_indices = np.random.choice(100, 20, replace=False)
    y_pred[noise_indices] = np.random.randint(0, 3, 20)
    
    # Evaluate
    eval_results = evaluator.evaluate_classification(y_true, y_pred, class_names=['A', 'B', 'C'])
    print(f"Classification accuracy: {eval_results['accuracy']:.3f}")
    print(f"F1 score: {eval_results['f1_score']:.3f}")
    
    # Test cross-validation (mock)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    
    cv_results = evaluator.cross_validate_model(model, X, y)
    print(f"Cross-validation mean score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    print("\n✅ All research methodology tools tested successfully!")