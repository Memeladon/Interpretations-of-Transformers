# Quick Start Guide

Get started with Interpretations-of-Transformers in 5 minutes!

## Installation

### Option 1: Install from source

```bash
git clone https://github.com/Memeladon/Interpretations-of-Transformers.git
cd Interpretations-of-Transformers
pip install -r requirements.txt
```

### Option 2: Install as package

```bash
git clone https://github.com/Memeladon/Interpretations-of-Transformers.git
cd Interpretations-of-Transformers
pip install -e .
```

## Quick Examples

### 1. Analyze a Single Text

```python
from src.embedding_interpreter import TransformerEmbeddingInterpreter

# Initialize
interpreter = TransformerEmbeddingInterpreter()

# Analyze
text = "Natural language processing is fascinating!"
results = interpreter.interpret(text)

# View results
print(f"Formality: {results['style_analysis']['formality']['mean_formality']:.3f}")
print(f"Sentiment: {results['style_analysis']['sentiment']['polarity']}")
```

### 2. Compare Multiple Texts

```python
texts = [
    "The research demonstrates significant improvements.",
    "This AI stuff is really cool!"
]

# Get comparative analysis
results = interpreter.interpret(texts)

# Check style differences
for i, score in enumerate(results['style_analysis']['formality']['scores']):
    print(f"Text {i+1} formality: {score:.3f}")
```

### 3. Semantic Clustering

```python
texts = [
    "Machine learning uses neural networks.",
    "Deep learning is a subset of AI.",
    "Cats and dogs are pets.",
    "Birds can fly in the sky."
]

results = interpreter.analyze_semantics(texts)

# See how texts cluster
clusters = results['clusters']
print(f"Found {clusters['n_clusters']} semantic groups")
print(f"Cluster labels: {clusters['labels']}")
```

### 4. Find Similar Texts

```python
new_text = "Artificial intelligence is transforming society."

reference_texts = [
    "Machine learning is part of AI research.",
    "The weather is nice today.",
    "Neural networks process information."
]

# Compare with references
results = interpreter.analyze_semantics(
    [new_text], 
    reference_texts=reference_texts
)

# Get most similar reference
comparison = results['reference_comparison']['comparisons'][0]
print(f"Most similar: {comparison['most_similar_reference']}")
print(f"Similarity: {comparison['similarity_score']:.3f}")
```

### 5. Visualize Results

```python
from src.visualization import plot_comprehensive_analysis

texts = [
    "Academic research paper abstract.",
    "Casual social media post!",
    "Technical documentation example.",
    "Everyday conversation text."
]

results = interpreter.interpret(texts)

# Create comprehensive visualization
plot_comprehensive_analysis(
    results,
    text_labels=['Academic', 'Casual', 'Technical', 'Everyday']
)
```

### 6. Use Command Line

```bash
# Analyze from command line
python -m src --text "Your text here" --model bert-base-uncased

# Multiple texts
python -m src --texts "First text" "Second text" "Third text"

# Style analysis only
python -m src --text "Your text" --style-only

# Semantic analysis only
python -m src --text "Your text" --semantic-only
```

### 7. Run the Example

```bash
# Run the comprehensive example
python examples/basic_usage.py

# Run validation tests
python examples/validate.py
```

### 8. Use in Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open the demo notebook
# examples/interpretation_demo.ipynb
```

## Common Use Cases

### Detect Writing Style

```python
# Detect if text is formal or informal
result = interpreter.analyze_style(["Your text here"])
formality = result['formality']['mean_formality']

if formality > 0.6:
    print("Formal writing")
elif formality > 0.3:
    print("Neutral writing")
else:
    print("Informal writing")
```

### Check Semantic Similarity

```python
# Check if two texts are semantically similar
similarity = interpreter.compare_embeddings(
    ["First text"],
    ["Second text"]
)

if similarity[0][0] > 0.8:
    print("Very similar")
elif similarity[0][0] > 0.5:
    print("Somewhat similar")
else:
    print("Different topics")
```

### Analyze Sentiment

```python
result = interpreter.analyze_style(["Your text"])
sentiment = result['sentiment']['polarity']
print(f"Sentiment: {sentiment}")  # positive, negative, or neutral
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [examples/basic_usage.py](examples/basic_usage.py) for more examples
- Try the [Jupyter notebook](examples/interpretation_demo.ipynb) for interactive exploration
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Troubleshooting

### Import Error: No module named 'torch'

Install PyTorch:
```bash
pip install torch
```

### Import Error: No module named 'transformers'

Install Transformers:
```bash
pip install transformers
```

### Model Download Issues

If model download is slow, you can:
1. Use a smaller model: `model_name="distilbert-base-uncased"`
2. Download models manually and load from local path

### Memory Issues

Use a smaller model or reduce batch size:
```python
interpreter = TransformerEmbeddingInterpreter(
    model_name="distilbert-base-uncased"
)
```

## Support

- Open an issue on GitHub for bug reports
- Check existing issues for solutions
- Read the documentation for detailed information

Happy interpreting! 🚀
