"""
Transformer Embedding Interpretation Package

This package provides tools for interpreting transformer embeddings
at the level of style and semantics.
"""

from .embedding_interpreter import TransformerEmbeddingInterpreter
from .style_analyzer import StyleAnalyzer
from .semantic_analyzer import SemanticAnalyzer

__all__ = [
    'TransformerEmbeddingInterpreter',
    'StyleAnalyzer',
    'SemanticAnalyzer',
]

__version__ = '0.1.0'
