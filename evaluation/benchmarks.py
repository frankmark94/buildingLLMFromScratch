#!/usr/bin/env python3
"""
Downstream task evaluation and benchmarking for language models.
Includes text classification, generation quality, and reasoning tasks.
"""

import re
import json
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class TextClassificationBenchmark:
    """Benchmark for text classification tasks."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_sentiment_classification(self, test_data: List[Tuple[str, int]],
                                        max_length: int = 512) -> Dict[str, float]:
        """Evaluate sentiment classification using zero-shot or few-shot prompting."""
        
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            for text, label in tqdm(test_data, desc="Sentiment classification"):
                # Create prompt for sentiment classification
                prompt = f"Text: {text}\nSentiment: "
                
                # Tokenize
                if hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(prompt)
                else:
                    tokens = self.tokenizer.tokenize_text(prompt)
                
                if len(tokens) > max_length - 10:  # Leave room for generation
                    tokens = tokens[:max_length - 10]
                
                input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Generate response
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False
                )
                
                # Extract generated tokens
                new_tokens = generated[0, input_ids.size(1):].cpu().tolist()
                
                # Decode response
                if hasattr(self.tokenizer, 'decode'):
                    response = self.tokenizer.decode(new_tokens)
                else:
                    response = self.tokenizer.decode_tokens(new_tokens)
                
                # Parse sentiment prediction
                pred_label = self._parse_sentiment_response(response)
                
                if pred_label is not None:
                    predictions.append(pred_label)
                    true_labels.append(label)
                    
                    if pred_label == label:
                        correct += 1
                    total += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        
        if len(predictions) > 0 and len(set(predictions)) > 1:
            f1 = f1_score(true_labels, predictions, average='weighted')
            precision, recall, _, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted'
            )
        else:
            f1 = precision = recall = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'total_samples': total
        }
    
    def _parse_sentiment_response(self, response: str) -> Optional[int]:
        """Parse sentiment from model response."""
        response = response.lower().strip()
        
        # Simple parsing - can be made more sophisticated
        if 'positive' in response or 'good' in response or 'happy' in response:
            return 1
        elif 'negative' in response or 'bad' in response or 'sad' in response:
            return 0
        else:
            # Try to extract first word
            words = response.split()
            if words:
                first_word = words[0].strip('.,!?')
                if first_word in ['positive', 'good', 'happy', '1']:
                    return 1
                elif first_word in ['negative', 'bad', 'sad', '0']:
                    return 0
        
        return None


class GenerationQualityBenchmark:
    """Benchmark for text generation quality."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_completion_quality(self, prompts: List[str], 
                                   reference_completions: List[str],
                                   max_new_tokens: int = 100) -> Dict[str, float]:
        """Evaluate completion quality against references."""
        
        bleu_scores = []
        rouge_scores = []
        
        self.model.eval()
        
        with torch.no_grad():
            for prompt, reference in tqdm(zip(prompts, reference_completions), 
                                        desc="Evaluating completions"):
                # Generate completion
                generated_text = self._generate_completion(prompt, max_new_tokens)
                
                # Calculate BLEU score (simple approximation)
                bleu = self._calculate_bleu(generated_text, reference)
                bleu_scores.append(bleu)
                
                # Calculate ROUGE score (simple approximation)
                rouge = self._calculate_rouge(generated_text, reference)
                rouge_scores.append(rouge)
        
        return {
            'average_bleu': np.mean(bleu_scores),
            'average_rouge': np.mean(rouge_scores),
            'bleu_scores': bleu_scores,
            'rouge_scores': rouge_scores
        }
    
    def evaluate_coherence(self, prompts: List[str], 
                          max_new_tokens: int = 200) -> Dict[str, float]:
        """Evaluate text coherence using perplexity-based metrics."""
        
        coherence_scores = []
        repetition_scores = []
        
        self.model.eval()
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Evaluating coherence"):
                generated_text = self._generate_completion(prompt, max_new_tokens)
                
                # Calculate coherence (inverse perplexity)
                coherence = self._calculate_coherence(generated_text)
                coherence_scores.append(coherence)
                
                # Calculate repetition penalty
                repetition = self._calculate_repetition(generated_text)
                repetition_scores.append(repetition)
        
        return {
            'average_coherence': np.mean(coherence_scores),
            'average_repetition': np.mean(repetition_scores),
            'coherence_scores': coherence_scores,
            'repetition_scores': repetition_scores
        }
    
    def _generate_completion(self, prompt: str, max_new_tokens: int) -> str:
        """Generate text completion for a prompt."""
        # Tokenize prompt
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer.tokenize_text(prompt)
        
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
        
        # Extract and decode new tokens
        new_tokens = generated[0, input_ids.size(1):].cpu().tolist()
        
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(new_tokens)
        else:
            return self.tokenizer.decode_tokens(new_tokens)
    
    def _calculate_bleu(self, generated: str, reference: str) -> float:
        """Simple BLEU score approximation."""
        generated_words = generated.lower().split()
        reference_words = reference.lower().split()
        
        if not generated_words or not reference_words:
            return 0.0
        
        # Count matching words (simple unigram BLEU)
        matches = sum(1 for word in generated_words if word in reference_words)
        precision = matches / len(generated_words)
        
        # Length penalty
        brevity_penalty = min(1.0, len(generated_words) / len(reference_words))
        
        return precision * brevity_penalty
    
    def _calculate_rouge(self, generated: str, reference: str) -> float:
        """Simple ROUGE score approximation."""
        generated_words = set(generated.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 0.0
        
        intersection = generated_words.intersection(reference_words)
        return len(intersection) / len(reference_words)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score based on model perplexity."""
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer.tokenize_text(text)
        
        if len(tokens) <= 1:
            return 0.0
        
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs['loss'].item()
            
        return 1.0 / (1.0 + loss)  # Convert to coherence score
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition score (lower is better)."""
        words = text.lower().split()
        
        if len(words) <= 1:
            return 0.0
        
        # Count word repetitions
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate repetition ratio
        total_words = len(words)
        unique_words = len(word_counts)
        repetition_ratio = 1.0 - (unique_words / total_words)
        
        return repetition_ratio


class ReasoningBenchmark:
    """Benchmark for reasoning and problem-solving tasks."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_arithmetic(self, problems: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate arithmetic reasoning capability."""
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for problem, answer in tqdm(problems, desc="Arithmetic evaluation"):
                prompt = f"Problem: {problem}\nAnswer: "
                
                # Generate answer
                generated_answer = self._generate_short_response(prompt, max_tokens=10)
                
                # Check if answer is correct
                if self._compare_arithmetic_answers(generated_answer, answer):
                    correct += 1
                
                total += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total
        }
    
    def evaluate_common_sense(self, questions: List[Tuple[str, List[str], int]]) -> Dict[str, float]:
        """Evaluate common sense reasoning (multiple choice)."""
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for question, choices, correct_idx in tqdm(questions, desc="Common sense evaluation"):
                # Format as multiple choice
                prompt = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65 + i)}. {choice}\n"
                prompt += "Answer: "
                
                # Generate answer
                response = self._generate_short_response(prompt, max_tokens=5)
                
                # Parse choice
                predicted_idx = self._parse_multiple_choice(response, len(choices))
                
                if predicted_idx == correct_idx:
                    correct += 1
                
                total += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total
        }
    
    def _generate_short_response(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate short response for reasoning tasks."""
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer.tokenize_text(prompt)
        
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False
        )
        
        new_tokens = generated[0, input_ids.size(1):].cpu().tolist()
        
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(new_tokens)
        else:
            return self.tokenizer.decode_tokens(new_tokens)
    
    def _compare_arithmetic_answers(self, generated: str, expected: str) -> bool:
        """Compare arithmetic answers."""
        # Extract numbers from both strings
        generated_nums = re.findall(r'-?\d+\.?\d*', generated.strip())
        expected_nums = re.findall(r'-?\d+\.?\d*', expected.strip())
        
        if generated_nums and expected_nums:
            try:
                gen_val = float(generated_nums[0])
                exp_val = float(expected_nums[0])
                return abs(gen_val - exp_val) < 1e-6
            except ValueError:
                pass
        
        # Fallback to string comparison
        return generated.strip().lower() == expected.strip().lower()
    
    def _parse_multiple_choice(self, response: str, num_choices: int) -> Optional[int]:
        """Parse multiple choice answer from response."""
        response = response.strip().upper()
        
        # Look for letter answers (A, B, C, D, etc.)
        for i in range(num_choices):
            if chr(65 + i) in response:
                return i
        
        # Look for number answers (1, 2, 3, 4, etc.)
        numbers = re.findall(r'\d+', response)
        if numbers:
            try:
                num = int(numbers[0])
                if 1 <= num <= num_choices:
                    return num - 1
            except ValueError:
                pass
        
        return None


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite combining multiple evaluation tasks."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.classification_benchmark = TextClassificationBenchmark(model, tokenizer, device)
        self.generation_benchmark = GenerationQualityBenchmark(model, tokenizer, device)
        self.reasoning_benchmark = ReasoningBenchmark(model, tokenizer, device)
    
    def run_full_evaluation(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive evaluation across all benchmarks."""
        
        results = {}
        
        # Text classification
        if 'sentiment_data' in test_data:
            logger.info("Running sentiment classification benchmark...")
            sentiment_results = self.classification_benchmark.evaluate_sentiment_classification(
                test_data['sentiment_data']
            )
            results['sentiment_classification'] = sentiment_results
        
        # Text generation
        if 'completion_data' in test_data:
            logger.info("Running text completion benchmark...")
            completion_results = self.generation_benchmark.evaluate_completion_quality(
                test_data['completion_data']['prompts'],
                test_data['completion_data']['references']
            )
            results['text_completion'] = completion_results
        
        if 'coherence_prompts' in test_data:
            logger.info("Running coherence benchmark...")
            coherence_results = self.generation_benchmark.evaluate_coherence(
                test_data['coherence_prompts']
            )
            results['coherence'] = coherence_results
        
        # Reasoning
        if 'arithmetic_problems' in test_data:
            logger.info("Running arithmetic reasoning benchmark...")
            arithmetic_results = self.reasoning_benchmark.evaluate_arithmetic(
                test_data['arithmetic_problems']
            )
            results['arithmetic_reasoning'] = arithmetic_results
        
        if 'commonsense_questions' in test_data:
            logger.info("Running common sense reasoning benchmark...")
            commonsense_results = self.reasoning_benchmark.evaluate_common_sense(
                test_data['commonsense_questions']
            )
            results['commonsense_reasoning'] = commonsense_results
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        results['overall_score'] = overall_score
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall benchmark score."""
        scores = []
        
        # Extract accuracy/score from each benchmark
        for task_name, task_results in results.items():
            if isinstance(task_results, dict):
                if 'accuracy' in task_results:
                    scores.append(task_results['accuracy'])
                elif 'average_bleu' in task_results:
                    scores.append(task_results['average_bleu'])
                elif 'average_coherence' in task_results:
                    scores.append(task_results['average_coherence'])
        
        return np.mean(scores) if scores else 0.0


def create_sample_benchmark_data() -> Dict[str, Any]:
    """Create sample benchmark data for testing."""
    
    # Sample sentiment classification data
    sentiment_data = [
        ("This movie was amazing!", 1),
        ("I hated this book.", 0),
        ("The weather is nice today.", 1),
        ("This is terrible service.", 0),
        ("I love programming!", 1)
    ]
    
    # Sample completion data
    completion_data = {
        'prompts': [
            "The capital of France is",
            "Python is a programming language that",
            "The main ingredient in pizza is"
        ],
        'references': [
            "Paris",
            "is widely used for web development, data science, and artificial intelligence",
            "dough, sauce, and cheese"
        ]
    }
    
    # Sample coherence prompts
    coherence_prompts = [
        "Tell me about the history of computers.",
        "Explain how photosynthesis works.",
        "Describe the process of making coffee."
    ]
    
    # Sample arithmetic problems
    arithmetic_problems = [
        ("What is 15 + 27?", "42"),
        ("What is 8 * 9?", "72"),
        ("What is 100 - 37?", "63")
    ]
    
    # Sample common sense questions
    commonsense_questions = [
        ("What do you use to cut paper?", ["Scissors", "Hammer", "Spoon"], 0),
        ("Where do fish live?", ["Trees", "Water", "Sky"], 1),
        ("What season comes after spring?", ["Winter", "Summer", "Fall"], 1)
    ]
    
    return {
        'sentiment_data': sentiment_data,
        'completion_data': completion_data,
        'coherence_prompts': coherence_prompts,
        'arithmetic_problems': arithmetic_problems,
        'commonsense_questions': commonsense_questions
    }


def save_benchmark_results(results: Dict[str, Any], filepath: str):
    """Save benchmark results to file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Benchmark results saved to {filepath}")


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results