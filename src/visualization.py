"""
Visualization Utilities for Embedding Interpretations

This module provides functions to visualize embedding interpretations,
including style and semantic analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional


def plot_style_comparison(
    style_results: Dict[str, Any],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot style analysis results.
    
    Args:
        style_results: Results from StyleAnalyzer
        labels: Optional labels for each text
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Style Analysis Results', fontsize=16, fontweight='bold')
    
    # Formality scores
    if 'formality' in style_results:
        ax = axes[0, 0]
        scores = style_results['formality']['scores']
        x = range(len(scores))
        ax.bar(x, scores, color='steelblue', alpha=0.7)
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Formality Score')
        ax.set_title('Formality Analysis')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Complexity scores
    if 'complexity' in style_results:
        ax = axes[0, 1]
        scores = style_results['complexity']['scores']
        x = range(len(scores))
        ax.bar(x, scores, color='coral', alpha=0.7)
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Complexity Score')
        ax.set_title('Complexity Analysis')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Sentiment scores
    if 'sentiment' in style_results:
        ax = axes[1, 0]
        scores = style_results['sentiment']['scores']
        x = range(len(scores))
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores]
        ax.bar(x, scores, color=colors, alpha=0.7)
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Analysis')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-1, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Style space (PCA)
    if 'style_dimensions' in style_results:
        ax = axes[1, 1]
        components = np.array(style_results['style_dimensions']['components'])
        if components.shape[1] >= 2:
            ax.scatter(components[:, 0], components[:, 1], 
                      s=100, alpha=0.6, c=range(len(components)), cmap='viridis')
            
            if labels:
                for i, label in enumerate(labels):
                    ax.annotate(label, (components[i, 0], components[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Style Dimension 1')
            ax.set_ylabel('Style Dimension 2')
            ax.set_title('Style Space (PCA)')
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_semantic_space(
    semantic_results: Dict[str, Any],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot semantic space visualization.
    
    Args:
        semantic_results: Results from SemanticAnalyzer
        labels: Optional labels for each text
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Semantic Analysis Results', fontsize=16, fontweight='bold')
    
    # Semantic space
    if 'semantic_space' in semantic_results:
        ax = axes[0]
        coords = np.array(semantic_results['semantic_space']['coordinates'])
        
        # Color by cluster if available
        if 'clusters' in semantic_results:
            cluster_labels = semantic_results['clusters']['labels']
            scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                               c=cluster_labels, s=100, alpha=0.6, cmap='tab10')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(coords[:, 0], coords[:, 1], 
                      s=100, alpha=0.6, c='steelblue')
        
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (coords[i, 0], coords[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Semantic Dimension 1')
        ax.set_ylabel('Semantic Dimension 2')
        ax.set_title('Semantic Space Visualization')
        ax.grid(alpha=0.3)
    
    # Similarity matrix
    if 'similarity_matrix' in semantic_results:
        ax = axes[1]
        matrix = np.array(semantic_results['similarity_matrix']['matrix'])
        
        im = ax.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1)
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Text Index')
        ax.set_title('Semantic Similarity Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # Add text annotations for small matrices
        if matrix.shape[0] <= 10:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_comprehensive_analysis(
    interpretation_results: Dict[str, Any],
    text_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comprehensive interpretation results.
    
    Args:
        interpretation_results: Results from interpret() method
        text_labels: Optional short labels for texts
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Comprehensive Embedding Interpretation', 
                 fontsize=18, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract results
    style_results = interpretation_results.get('style_analysis', {})
    semantic_results = interpretation_results.get('semantic_analysis', {})
    
    # 1. Formality
    if 'formality' in style_results:
        ax = fig.add_subplot(gs[0, 0])
        scores = style_results['formality']['scores']
        ax.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
        ax.set_title('Formality')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # 2. Complexity
    if 'complexity' in style_results:
        ax = fig.add_subplot(gs[0, 1])
        scores = style_results['complexity']['scores']
        ax.bar(range(len(scores)), scores, color='coral', alpha=0.7)
        ax.set_title('Complexity')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # 3. Sentiment
    if 'sentiment' in style_results:
        ax = fig.add_subplot(gs[0, 2])
        scores = style_results['sentiment']['scores']
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores]
        ax.bar(range(len(scores)), scores, color=colors, alpha=0.7)
        ax.set_title('Sentiment')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-1, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # 4. Style Space
    if 'style_dimensions' in style_results:
        ax = fig.add_subplot(gs[1, :2])
        components = np.array(style_results['style_dimensions']['components'])
        if components.shape[1] >= 2:
            scatter = ax.scatter(components[:, 0], components[:, 1], 
                               s=100, alpha=0.6, c=range(len(components)), cmap='viridis')
            if text_labels:
                for i, label in enumerate(text_labels):
                    ax.annotate(label, (components[i, 0], components[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax.set_xlabel('Style Dimension 1')
            ax.set_ylabel('Style Dimension 2')
            ax.set_title('Style Space (PCA)')
            ax.grid(alpha=0.3)
    
    # 5. Semantic Space
    if 'semantic_space' in semantic_results:
        ax = fig.add_subplot(gs[1, 2])
        coords = np.array(semantic_results['semantic_space']['coordinates'])
        
        if 'clusters' in semantic_results:
            cluster_labels = semantic_results['clusters']['labels']
            scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                               c=cluster_labels, s=100, alpha=0.6, cmap='tab10')
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6, c='steelblue')
        
        ax.set_xlabel('Semantic Dim 1')
        ax.set_ylabel('Semantic Dim 2')
        ax.set_title('Semantic Space')
        ax.grid(alpha=0.3)
    
    # 6. Similarity Matrix
    if 'similarity_matrix' in semantic_results:
        ax = fig.add_subplot(gs[2, :2])
        matrix = np.array(semantic_results['similarity_matrix']['matrix'])
        im = ax.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1)
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Text Index')
        ax.set_title('Semantic Similarity Matrix')
        plt.colorbar(im, ax=ax, label='Similarity')
    
    # 7. Summary Statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    summary_text = "Summary Statistics\n" + "=" * 25 + "\n\n"
    
    if 'formality' in style_results:
        summary_text += f"Avg Formality: {style_results['formality']['mean_formality']:.3f}\n"
    if 'complexity' in style_results:
        summary_text += f"Avg Complexity: {style_results['complexity']['mean_complexity']:.3f}\n"
    if 'coherence' in semantic_results:
        summary_text += f"Coherence: {semantic_results['coherence']['coherence_score']:.3f}\n"
    if 'clusters' in semantic_results:
        summary_text += f"Clusters: {semantic_results['clusters']['n_clusters']}\n"
    
    summary_text += f"\nModel: {interpretation_results.get('model', 'N/A')}"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
           family='monospace')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
