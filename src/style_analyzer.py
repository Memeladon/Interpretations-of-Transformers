"""
Style Analyzer Module

This module analyzes stylistic features in transformer embeddings,
including formality, complexity, sentiment, and writing style patterns.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class StyleAnalyzer:
    """
    Analyzer for extracting and interpreting stylistic features
    from transformer embeddings.
    """
    
    def __init__(self):
        """Initialize the Style Analyzer."""
        self.scaler = StandardScaler()
        self.pca = None
        
    def analyze(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze stylistic aspects of text embeddings.
        
        Args:
            texts: Input texts
            embeddings: Corresponding embeddings
            aspects: Specific aspects to analyze
            
        Returns:
            Dictionary with style analysis results
        """
        results = {}
        
        # Default aspects if none specified
        if aspects is None:
            aspects = ['formality', 'complexity', 'sentiment', 'variability']
        
        # Compute basic statistical features
        results['embedding_stats'] = self._compute_embedding_stats(embeddings)
        
        # Analyze each requested aspect
        if 'formality' in aspects:
            results['formality'] = self._analyze_formality(texts, embeddings)
            
        if 'complexity' in aspects:
            results['complexity'] = self._analyze_complexity(texts, embeddings)
            
        if 'sentiment' in aspects:
            results['sentiment'] = self._analyze_sentiment(texts, embeddings)
            
        if 'variability' in aspects:
            results['variability'] = self._analyze_variability(embeddings)
            
        # Dimensionality reduction for visualization
        results['style_dimensions'] = self._reduce_dimensions(embeddings)
        
        return results
    
    def _compute_embedding_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistics on embeddings."""
        return {
            'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean()),
            'std_norm': float(np.linalg.norm(embeddings, axis=1).std()),
            'dimensionality': embeddings.shape[1],
            'num_samples': embeddings.shape[0],
            'mean_activation': float(embeddings.mean()),
            'std_activation': float(embeddings.std()),
        }
    
    def _analyze_formality(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze formality based on text features and embedding patterns.
        
        This uses heuristics based on:
        - Average word length
        - Sentence length
        - Presence of contractions
        - Embedding magnitude patterns
        """
        formality_scores = []
        
        for text in texts:
            # Text-based features
            words = text.split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            
            # Simple heuristic: longer words often indicate higher formality
            # Normalize to 0-1 scale (assuming avg word length 3-8)
            formality = min(max((avg_word_len - 3) / 5, 0), 1)
            
            # Check for contractions (informal)
            contractions = ["n't", "'ll", "'ve", "'re", "'m", "'d"]
            has_contractions = any(c in text for c in contractions)
            if has_contractions:
                formality *= 0.8
                
            formality_scores.append(formality)
        
        return {
            'scores': formality_scores,
            'mean_formality': float(np.mean(formality_scores)),
            'std_formality': float(np.std(formality_scores)),
        }
    
    def _analyze_complexity(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze linguistic complexity.
        
        Based on:
        - Sentence length
        - Vocabulary diversity (unique words ratio)
        - Embedding variance
        """
        complexity_scores = []
        
        for text, emb in zip(texts, embeddings):
            words = text.split()
            
            # Length-based complexity
            length_complexity = min(len(words) / 50, 1.0)  # Normalize
            
            # Vocabulary diversity
            unique_ratio = len(set(words)) / len(words) if words else 0
            
            # Embedding variance (higher variance may indicate complexity)
            emb_variance = np.var(emb)
            emb_complexity = min(emb_variance * 10, 1.0)  # Scale
            
            # Combined score
            complexity = (length_complexity + unique_ratio + emb_complexity) / 3
            complexity_scores.append(complexity)
        
        return {
            'scores': complexity_scores,
            'mean_complexity': float(np.mean(complexity_scores)),
            'std_complexity': float(np.std(complexity_scores)),
        }
    
    def _analyze_sentiment(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze sentiment polarity based on embedding patterns.
        
        This is a simplified heuristic approach. For production,
        consider using a dedicated sentiment model.
        """
        # Simple lexicon-based approach
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'love', 'best', 'fantastic', 'perfect', 'beautiful'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 
                         'horrible', 'poor', 'disappointing', 'ugly', 'sad'}
        
        sentiment_scores = []
        
        for text in texts:
            words = set(text.lower().split())
            
            pos_count = len(words.intersection(positive_words))
            neg_count = len(words.intersection(negative_words))
            
            # Sentiment score: -1 (negative) to 1 (positive)
            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0
                
            sentiment_scores.append(sentiment)
        
        return {
            'scores': sentiment_scores,
            'mean_sentiment': float(np.mean(sentiment_scores)),
            'polarity': 'positive' if np.mean(sentiment_scores) > 0.2 
                       else 'negative' if np.mean(sentiment_scores) < -0.2 
                       else 'neutral',
        }
    
    def _analyze_variability(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Analyze variability in writing style across samples.
        """
        if len(embeddings) < 2:
            return {
                'pairwise_distances': [],
                'mean_distance': 0.0,
                'consistency_score': 1.0,
            }
        
        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(float(dist))
        
        mean_dist = np.mean(distances)
        
        # Consistency score (inverse of variability)
        # Normalize assuming typical distances are 0-10
        consistency = max(0, 1 - mean_dist / 10)
        
        return {
            'pairwise_distances': distances[:10],  # Limit for readability
            'mean_distance': float(mean_dist),
            'consistency_score': float(consistency),
        }
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2
    ) -> Dict[str, Any]:
        """
        Reduce embedding dimensions for visualization.
        """
        if len(embeddings) < n_components:
            return {
                'components': embeddings.tolist(),
                'explained_variance': [],
            }
        
        # Standardize
        scaled = self.scaler.fit_transform(embeddings)
        
        # PCA
        self.pca = PCA(n_components=n_components)
        reduced = self.pca.fit_transform(scaled)
        
        return {
            'components': reduced.tolist(),
            'explained_variance': self.pca.explained_variance_ratio_.tolist(),
            'n_components': n_components,
        }
