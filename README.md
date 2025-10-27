# Interpretations-of-Transformers

Development of a method for interpreting transformer embeddings at the level of style and semantics.

## Overview

This project provides a comprehensive framework for interpreting transformer model embeddings by analyzing both **stylistic** and **semantic** aspects of text representations. The framework enables researchers and practitioners to gain deeper insights into how transformer models encode linguistic information.

## Features

### Style Analysis
- **Formality Detection**: Analyze the level of formality in text based on word choice and sentence structure
- **Complexity Measurement**: Assess linguistic complexity through vocabulary diversity and sentence patterns
- **Sentiment Analysis**: Extract sentiment polarity from text representations
- **Style Consistency**: Measure variability in writing style across multiple texts
- **Dimensionality Reduction**: Visualize style patterns in reduced dimensional space (PCA)

### Semantic Analysis
- **Similarity Computation**: Calculate semantic similarity between texts using cosine similarity
- **Topic Clustering**: Automatically cluster texts into semantic groups using K-means
- **Coherence Analysis**: Measure semantic coherence within a collection of texts
- **Concept Density**: Assess how tightly packed concepts are in the semantic space
- **Reference Comparison**: Compare input texts against reference texts semantically
- **Semantic Space Visualization**: Visualize semantic relationships using t-SNE

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Memeladon/Interpretations-of-Transformers.git
cd Interpretations-of-Transformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.embedding_interpreter import TransformerEmbeddingInterpreter

# Initialize the interpreter
interpreter = TransformerEmbeddingInterpreter(model_name="bert-base-uncased")

# Analyze style
texts = [
    "The research demonstrates significant improvements.",
    "This AI stuff is really cool!"
]

style_results = interpreter.analyze_style(texts)
print(f"Formality: {style_results['formality']['mean_formality']:.3f}")

# Analyze semantics
semantic_results = interpreter.analyze_semantics(texts)
print(f"Coherence: {semantic_results['coherence']['coherence_score']:.3f}")

# Comprehensive interpretation
results = interpreter.interpret(texts, include_style=True, include_semantics=True)
```

## Usage Examples

### Basic Usage

```python
from src.embedding_interpreter import TransformerEmbeddingInterpreter

# Initialize with default BERT model
interpreter = TransformerEmbeddingInterpreter()

# Or use a different model
interpreter = TransformerEmbeddingInterpreter(model_name="roberta-base")

# Analyze a single text
text = "Natural language processing is fascinating."
results = interpreter.interpret(text)
```

### Style Analysis

```python
# Analyze specific style aspects
texts = ["Formal academic text.", "Casual everyday language!"]
style_results = interpreter.analyze_style(
    texts, 
    aspects=['formality', 'complexity', 'sentiment']
)

# Access individual metrics
formality_scores = style_results['formality']['scores']
complexity_scores = style_results['complexity']['scores']
sentiment_scores = style_results['sentiment']['scores']
```

### Semantic Analysis

```python
# Analyze semantic relationships
texts = [
    "Machine learning uses neural networks.",
    "Deep learning is a subset of AI.",
    "Cats and dogs are popular pets."
]

semantic_results = interpreter.analyze_semantics(texts)

# Check clustering results
n_clusters = semantic_results['clusters']['n_clusters']
labels = semantic_results['clusters']['labels']

# Check coherence
coherence = semantic_results['coherence']['coherence_score']
```

### Comparison with Reference Texts

```python
# Compare texts against references
input_texts = ["New text to analyze."]
reference_texts = [
    "Similar reference text.",
    "Different reference text."
]

results = interpreter.analyze_semantics(
    input_texts, 
    reference_texts=reference_texts
)

# Get similarity to most similar reference
comparison = results['reference_comparison']
```

### Embedding Extraction

```python
# Get raw embeddings
embeddings = interpreter.get_embeddings(texts)

# Use different pooling strategies
cls_embeddings = interpreter.get_embeddings(texts, pooling='cls')
max_embeddings = interpreter.get_embeddings(texts, pooling='max')

# Extract from specific layer
layer_embeddings = interpreter.get_embeddings(texts, layer=-2)
```

### Visualization

```python
from src.visualization import plot_comprehensive_analysis

# Create comprehensive visualization
results = interpreter.interpret(texts)
plot_comprehensive_analysis(
    results, 
    text_labels=['Text 1', 'Text 2', 'Text 3'],
    save_path='analysis_results.png'
)
```

## Running Examples

Run the basic usage example:

```bash
python examples/basic_usage.py
```

This will demonstrate:
- Style analysis on formal vs. informal texts
- Semantic analysis on technical texts
- Embedding comparison across different writing styles
- Comprehensive interpretation of mixed text collections

## Project Structure

```
Interpretations-of-Transformers/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── embedding_interpreter.py    # Main interpreter class
│   ├── style_analyzer.py           # Style analysis module
│   ├── semantic_analyzer.py        # Semantic analysis module
│   └── visualization.py            # Visualization utilities
├── examples/
│   └── basic_usage.py              # Example usage script
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## Core Components

### TransformerEmbeddingInterpreter

Main class for embedding interpretation:
- `get_embeddings()`: Extract embeddings from transformer models
- `analyze_style()`: Perform style analysis
- `analyze_semantics()`: Perform semantic analysis
- `interpret()`: Comprehensive interpretation combining both analyses
- `compare_embeddings()`: Compare embeddings between text sets

### StyleAnalyzer

Analyzes stylistic aspects:
- Formality detection based on word length and contractions
- Complexity measurement using sentence length and vocabulary diversity
- Sentiment analysis using lexicon-based approach
- Style variability and consistency metrics
- PCA-based dimensionality reduction for visualization

### SemanticAnalyzer

Analyzes semantic content:
- Cosine similarity computation
- K-means clustering for topic detection
- Coherence scoring based on centroid distance
- Concept density measurement
- t-SNE visualization of semantic space

## Supported Models

The framework supports any Hugging Face transformer model, including:
- BERT (`bert-base-uncased`, `bert-large-uncased`)
- RoBERTa (`roberta-base`, `roberta-large`)
- DistilBERT (`distilbert-base-uncased`)
- ALBERT (`albert-base-v2`)
- And many more from the Hugging Face model hub

## Method Details

### Style Analysis Methodology

1. **Formality**: Computed using average word length and presence of contractions
2. **Complexity**: Based on sentence length, vocabulary diversity, and embedding variance
3. **Sentiment**: Lexicon-based approach with positive/negative word matching
4. **Variability**: Pairwise distance computation in embedding space

### Semantic Analysis Methodology

1. **Similarity**: Cosine similarity in embedding space
2. **Clustering**: K-means with automatic cluster number detection
3. **Coherence**: Average distance to centroid in embedding space
4. **Density**: Mean pairwise distance using cosine metric
5. **Visualization**: t-SNE for 2D projection of semantic space

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy 1.24+
- scikit-learn 1.3+
- SciPy 1.11+
- Matplotlib 3.7+ (for visualization)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{transformer_embedding_interpretation,
  title={Interpretations of Transformers: A Method for Interpreting Transformer Embeddings},
  author={Memeladon},
  year={2024},
  url={https://github.com/Memeladon/Interpretations-of-Transformers}
}
```

## Future Enhancements

Potential areas for expansion:
- Integration with attention visualization
- Support for multilingual analysis
- Fine-grained aspect-based analysis
- Interactive web interface
- Pre-computed embedding caching
- Custom style/semantic metrics
- Advanced concept extraction methods