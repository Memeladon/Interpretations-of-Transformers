"""
Semantic Analyzer Module

This module analyzes semantic content and relationships in transformer embeddings,
including topic clustering, semantic similarity, and concept detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


class SemanticAnalyzer:
    """
    Analyzer for extracting and interpreting semantic features
    from transformer embeddings.
    """
    
    def __init__(self):
        """Initialize the Semantic Analyzer."""
        self.tsne = None
        self.kmeans = None
        
    def analyze(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        reference_embeddings: Optional[np.ndarray] = None,
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze semantic aspects of text embeddings.
        
        Args:
            texts: Input texts
            embeddings: Corresponding embeddings
            reference_embeddings: Optional reference embeddings for comparison
            reference_texts: Optional reference texts
            
        Returns:
            Dictionary with semantic analysis results
        """
        results = {}
        
        # Semantic similarity analysis
        results['similarity_matrix'] = self._compute_similarity_matrix(embeddings)
        
        # Topic clustering
        if len(embeddings) >= 2:
            results['clusters'] = self._cluster_topics(texts, embeddings)
        
        # Semantic coherence
        results['coherence'] = self._compute_coherence(embeddings)
        
        # Concept density (how concentrated the semantic space is)
        results['concept_density'] = self._compute_concept_density(embeddings)
        
        # Reference comparison if provided
        if reference_embeddings is not None and reference_texts is not None:
            results['reference_comparison'] = self._compare_to_reference(
                texts, embeddings, reference_texts, reference_embeddings
            )
        
        # Semantic space visualization
        if len(embeddings) >= 3:
            results['semantic_space'] = self._visualize_semantic_space(embeddings)
        
        return results
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute pairwise semantic similarity matrix.
        """
        # Cosine similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Exclude diagonal for mean computation
        n = len(sim_matrix)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            mean_similarity = float(sim_matrix[mask].mean())
            std_similarity = float(sim_matrix[mask].std())
        else:
            mean_similarity = 1.0
            std_similarity = 0.0
        
        return {
            'matrix': sim_matrix.tolist(),
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
        }
    
    def _cluster_topics(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Cluster texts into semantic topics.
        """
        # Auto-determine number of clusters (heuristic)
        if n_clusters is None:
            n_clusters = min(max(2, len(embeddings) // 3), 5)
        
        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, len(embeddings))
        
        if n_clusters < 2:
            return {
                'n_clusters': 1,
                'labels': [0] * len(texts),
                'cluster_centers': embeddings.mean(axis=0).tolist(),
            }
        
        # K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(embeddings)
        
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return {
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'cluster_centers': self.kmeans.cluster_centers_.tolist(),
            'inertia': float(self.kmeans.inertia_),
            'cluster_sizes': {k: len(v) for k, v in clusters.items()},
        }
    
    def _compute_coherence(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute semantic coherence of the text collection.
        
        Higher coherence indicates texts are semantically related.
        """
        if len(embeddings) < 2:
            return {
                'coherence_score': 1.0,
                'interpretation': 'single_sample',
            }
        
        # Compute centroid
        centroid = embeddings.mean(axis=0)
        
        # Compute average distance to centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        avg_distance = float(distances.mean())
        
        # Normalize to coherence score (0-1, higher is more coherent)
        # Assuming typical distances are 0-5
        coherence_score = max(0, 1 - avg_distance / 5)
        
        # Interpretation
        if coherence_score > 0.7:
            interpretation = 'high_coherence'
        elif coherence_score > 0.4:
            interpretation = 'moderate_coherence'
        else:
            interpretation = 'low_coherence'
        
        return {
            'coherence_score': float(coherence_score),
            'avg_distance_to_centroid': avg_distance,
            'interpretation': interpretation,
        }
    
    def _compute_concept_density(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute how densely packed the concepts are in semantic space.
        """
        if len(embeddings) < 2:
            return {
                'density_score': 1.0,
                'spread': 0.0,
            }
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        
        distances = pdist(embeddings, metric='cosine')
        
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        
        # Density is inverse of spread
        density = max(0, 1 - mean_dist)
        
        return {
            'density_score': float(density),
            'mean_pairwise_distance': mean_dist,
            'std_pairwise_distance': std_dist,
        }
    
    def _compare_to_reference(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        reference_texts: List[str],
        reference_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare input texts to reference texts semantically.
        """
        # Compute similarities between each input and each reference
        similarities = cosine_similarity(embeddings, reference_embeddings)
        
        # Find most similar reference for each input
        most_similar_indices = similarities.argmax(axis=1)
        max_similarities = similarities.max(axis=1)
        
        comparisons = []
        for i, (text, sim_idx, sim_score) in enumerate(
            zip(texts, most_similar_indices, max_similarities)
        ):
            comparisons.append({
                'input_text': text,
                'most_similar_reference': reference_texts[sim_idx],
                'similarity_score': float(sim_score),
            })
        
        return {
            'comparisons': comparisons,
            'mean_similarity_to_reference': float(max_similarities.mean()),
            'min_similarity': float(max_similarities.min()),
            'max_similarity': float(max_similarities.max()),
        }
    
    def _visualize_semantic_space(
        self,
        embeddings: np.ndarray,
        n_components: int = 2
    ) -> Dict[str, Any]:
        """
        Create a low-dimensional representation of the semantic space.
        """
        if len(embeddings) < 3:
            return {
                'coordinates': embeddings[:, :n_components].tolist(),
                'method': 'raw',
            }
        
        # Use t-SNE for better semantic space visualization
        perplexity = min(30, len(embeddings) - 1)
        
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
        
        reduced = self.tsne.fit_transform(embeddings)
        
        return {
            'coordinates': reduced.tolist(),
            'method': 'tsne',
            'n_components': n_components,
            'perplexity': perplexity,
        }
    
    def extract_key_concepts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Extract key concepts from the text collection.
        
        This is a placeholder for more sophisticated concept extraction.
        In a full implementation, this could use attention weights or
        gradient-based importance scores.
        """
        # Simple word frequency analysis as baseline
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = [w.lower() for w in text.split() if len(w) > 3]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(top_k)
        
        return {
            'top_concepts': [word for word, _ in top_words],
            'concept_frequencies': {word: count for word, count in top_words},
        }
