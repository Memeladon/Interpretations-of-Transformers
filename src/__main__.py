"""
Command-line interface for Interpretations-of-Transformers.

Usage:
    python -m src --help
    python -m src --text "Your text here"
"""

import argparse
import sys
from embedding_interpreter import TransformerEmbeddingInterpreter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interpret transformer embeddings at style and semantic levels"
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Text to analyze (use quotes for multiple words)'
    )
    
    parser.add_argument(
        '--texts',
        nargs='+',
        help='Multiple texts to analyze'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help='Transformer model to use (default: bert-base-uncased)'
    )
    
    parser.add_argument(
        '--style-only',
        action='store_true',
        help='Perform only style analysis'
    )
    
    parser.add_argument(
        '--semantic-only',
        action='store_true',
        help='Perform only semantic analysis'
    )
    
    args = parser.parse_args()
    
    # Determine texts to analyze
    if args.text:
        texts = [args.text]
    elif args.texts:
        texts = args.texts
    else:
        print("Error: Please provide text to analyze using --text or --texts")
        parser.print_help()
        return 1
    
    # Initialize interpreter
    print(f"Initializing interpreter with model: {args.model}")
    try:
        interpreter = TransformerEmbeddingInterpreter(model_name=args.model)
    except Exception as e:
        print(f"Error initializing interpreter: {e}")
        print("Note: Make sure torch and transformers are installed:")
        print("  pip install torch transformers")
        return 1
    
    # Determine what to analyze
    include_style = not args.semantic_only
    include_semantics = not args.style_only
    
    # Perform analysis
    print(f"\nAnalyzing {len(texts)} text(s)...")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    try:
        results = interpreter.interpret(
            texts,
            include_style=include_style,
            include_semantics=include_semantics
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    # Display results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    if include_style:
        print("\nStyle Analysis:")
        style = results['style_analysis']
        
        if 'formality' in style:
            print(f"  Formality Score: {style['formality']['mean_formality']:.3f}")
        
        if 'complexity' in style:
            print(f"  Complexity Score: {style['complexity']['mean_complexity']:.3f}")
        
        if 'sentiment' in style:
            print(f"  Sentiment: {style['sentiment']['polarity']}")
            print(f"  Sentiment Score: {style['sentiment']['mean_sentiment']:.3f}")
    
    if include_semantics:
        print("\nSemantic Analysis:")
        semantic = results['semantic_analysis']
        
        if 'coherence' in semantic:
            print(f"  Coherence Score: {semantic['coherence']['coherence_score']:.3f}")
            print(f"  Interpretation: {semantic['coherence']['interpretation']}")
        
        if 'clusters' in semantic:
            print(f"  Detected Clusters: {semantic['clusters']['n_clusters']}")
        
        if 'similarity_matrix' in semantic:
            print(f"  Mean Similarity: {semantic['similarity_matrix']['mean_similarity']:.3f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
