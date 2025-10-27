"""
Main Transformer Embedding Interpreter Module

This module provides the main interface for interpreting transformer embeddings
by analyzing both style and semantic aspects.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel


class TransformerEmbeddingInterpreter:
    """
    A comprehensive interpreter for transformer embeddings that analyzes
    both stylistic and semantic aspects of text representations.
    
    This class combines style and semantic analysis to provide interpretable
    insights into how transformer models represent text at different levels.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None
    ):
        """
        Initialize the Transformer Embedding Interpreter.
        
        Args:
            model_name: Name of the pretrained transformer model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        layer: int = -1,
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Extract embeddings from the transformer model.
        
        Args:
            texts: Input text(s) to embed
            layer: Which layer to extract embeddings from (-1 for last layer)
            pooling: Pooling strategy ('mean', 'cls', 'max')
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            
        # Select layer
        hidden_states = outputs.hidden_states[layer]
        
        # Apply pooling
        if pooling == 'mean':
            # Mean pooling (excluding padding tokens)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        elif pooling == 'cls':
            # Use CLS token
            embeddings = hidden_states[:, 0, :]
        elif pooling == 'max':
            # Max pooling
            embeddings = hidden_states.max(1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
            
        return embeddings.cpu().numpy()
    
    def analyze_style(
        self,
        texts: Union[str, List[str]],
        aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the stylistic aspects of the embeddings.
        
        Args:
            texts: Input text(s) to analyze
            aspects: Specific style aspects to analyze (e.g., ['formality', 'complexity'])
            
        Returns:
            Dictionary containing style analysis results
        """
        from .style_analyzer import StyleAnalyzer
        
        embeddings = self.get_embeddings(texts)
        analyzer = StyleAnalyzer()
        
        return analyzer.analyze(texts, embeddings, aspects)
    
    def analyze_semantics(
        self,
        texts: Union[str, List[str]],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the semantic aspects of the embeddings.
        
        Args:
            texts: Input text(s) to analyze
            reference_texts: Optional reference texts for comparison
            
        Returns:
            Dictionary containing semantic analysis results
        """
        from .semantic_analyzer import SemanticAnalyzer
        
        embeddings = self.get_embeddings(texts)
        analyzer = SemanticAnalyzer()
        
        if reference_texts:
            ref_embeddings = self.get_embeddings(reference_texts)
            return analyzer.analyze(texts, embeddings, ref_embeddings, reference_texts)
        else:
            return analyzer.analyze(texts, embeddings)
    
    def interpret(
        self,
        texts: Union[str, List[str]],
        include_style: bool = True,
        include_semantics: bool = True,
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive interpretation of embeddings.
        
        Args:
            texts: Input text(s) to interpret
            include_style: Whether to include style analysis
            include_semantics: Whether to include semantic analysis
            reference_texts: Optional reference texts for semantic comparison
            
        Returns:
            Dictionary containing full interpretation results
        """
        results = {
            'texts': texts if isinstance(texts, list) else [texts],
            'model': self.model_name
        }
        
        if include_style:
            results['style_analysis'] = self.analyze_style(texts)
            
        if include_semantics:
            results['semantic_analysis'] = self.analyze_semantics(texts, reference_texts)
            
        return results
    
    def compare_embeddings(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compare embeddings between two sets of texts.
        
        Args:
            texts1: First set of text(s)
            texts2: Second set of text(s)
            metric: Similarity metric ('cosine', 'euclidean')
            
        Returns:
            Similarity/distance matrix
        """
        emb1 = self.get_embeddings(texts1)
        emb2 = self.get_embeddings(texts2)
        
        if metric == 'cosine':
            # Cosine similarity
            norm1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
            norm2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
            return norm1 @ norm2.T
        elif metric == 'euclidean':
            # Euclidean distance
            return np.sqrt(((emb1[:, None] - emb2[None, :]) ** 2).sum(-1))
        else:
            raise ValueError(f"Unknown metric: {metric}")
