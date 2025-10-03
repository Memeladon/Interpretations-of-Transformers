"""
Example usage of the Transformer Embedding Interpreter

This script demonstrates how to use the interpretation methods
for analyzing transformer embeddings at style and semantic levels.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embedding_interpreter import TransformerEmbeddingInterpreter


def main():
    """Run example interpretations."""
    
    print("=" * 80)
    print("Transformer Embedding Interpretation Example")
    print("=" * 80)
    print()
    
    # Initialize interpreter
    print("Initializing interpreter with BERT model...")
    interpreter = TransformerEmbeddingInterpreter(model_name="bert-base-uncased")
    print("✓ Model loaded successfully")
    print()
    
    # Example texts with different styles and semantics
    formal_texts = [
        "The research demonstrates significant improvements in natural language processing.",
        "Our findings indicate a substantial enhancement in model performance.",
    ]
    
    informal_texts = [
        "This AI stuff is really cool and it's getting better every day!",
        "I can't believe how awesome these language models are becoming.",
    ]
    
    technical_texts = [
        "The transformer architecture utilizes self-attention mechanisms.",
        "Neural networks employ backpropagation for gradient descent optimization.",
    ]
    
    # Example 1: Style Analysis
    print("-" * 80)
    print("Example 1: Style Analysis")
    print("-" * 80)
    
    print("\nAnalyzing formal texts:")
    for text in formal_texts:
        print(f"  - {text}")
    
    formal_analysis = interpreter.analyze_style(formal_texts)
    print(f"\nFormality Score: {formal_analysis['formality']['mean_formality']:.3f}")
    print(f"Complexity Score: {formal_analysis['complexity']['mean_complexity']:.3f}")
    
    print("\nAnalyzing informal texts:")
    for text in informal_texts:
        print(f"  - {text}")
    
    informal_analysis = interpreter.analyze_style(informal_texts)
    print(f"\nFormality Score: {informal_analysis['formality']['mean_formality']:.3f}")
    print(f"Complexity Score: {informal_analysis['complexity']['mean_complexity']:.3f}")
    
    # Example 2: Semantic Analysis
    print("\n" + "-" * 80)
    print("Example 2: Semantic Analysis")
    print("-" * 80)
    
    print("\nAnalyzing technical texts:")
    for text in technical_texts:
        print(f"  - {text}")
    
    semantic_analysis = interpreter.analyze_semantics(technical_texts)
    print(f"\nSemantic Coherence: {semantic_analysis['coherence']['coherence_score']:.3f}")
    print(f"Interpretation: {semantic_analysis['coherence']['interpretation']}")
    
    # Example 3: Comparison
    print("\n" + "-" * 80)
    print("Example 3: Embedding Comparison")
    print("-" * 80)
    
    print("\nComparing formal vs. informal texts...")
    similarity = interpreter.compare_embeddings(formal_texts, informal_texts)
    print(f"\nCross-style similarity matrix:")
    print(f"  Mean similarity: {similarity.mean():.3f}")
    print(f"  Max similarity: {similarity.max():.3f}")
    print(f"  Min similarity: {similarity.min():.3f}")
    
    # Example 4: Comprehensive Interpretation
    print("\n" + "-" * 80)
    print("Example 4: Comprehensive Interpretation")
    print("-" * 80)
    
    mixed_texts = formal_texts + informal_texts
    print("\nAnalyzing mixed text collection:")
    for i, text in enumerate(mixed_texts, 1):
        print(f"  {i}. {text}")
    
    results = interpreter.interpret(
        mixed_texts,
        include_style=True,
        include_semantics=True
    )
    
    print("\nResults Summary:")
    print(f"  Number of texts analyzed: {len(results['texts'])}")
    print(f"  Model used: {results['model']}")
    print(f"  Mean formality: {results['style_analysis']['formality']['mean_formality']:.3f}")
    print(f"  Mean complexity: {results['style_analysis']['complexity']['mean_complexity']:.3f}")
    print(f"  Semantic coherence: {results['semantic_analysis']['coherence']['coherence_score']:.3f}")
    
    if 'clusters' in results['semantic_analysis']:
        n_clusters = results['semantic_analysis']['clusters']['n_clusters']
        print(f"  Detected semantic clusters: {n_clusters}")
    
    print("\n" + "=" * 80)
    print("Interpretation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
