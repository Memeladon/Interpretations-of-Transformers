"""
Simple validation test for the interpretation framework.

This script validates the basic structure and imports without downloading models.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    success = True
    
    try:
        import embedding_interpreter
        print("✓ embedding_interpreter imported")
    except ImportError as e:
        if 'torch' in str(e) or 'transformers' in str(e):
            print("⚠ embedding_interpreter requires torch/transformers (optional for validation)")
        else:
            print(f"✗ embedding_interpreter failed: {e}")
            success = False
    except Exception as e:
        print(f"✗ embedding_interpreter failed: {e}")
        success = False
    
    try:
        import style_analyzer
        print("✓ style_analyzer imported")
    except Exception as e:
        print(f"✗ style_analyzer failed: {e}")
        success = False
    
    try:
        import semantic_analyzer
        print("✓ semantic_analyzer imported")
    except Exception as e:
        print(f"✗ semantic_analyzer failed: {e}")
        success = False
    
    try:
        import visualization
        print("✓ visualization imported")
    except Exception as e:
        print(f"✗ visualization failed: {e}")
        success = False
    
    return success


def test_class_initialization():
    """Test that classes can be instantiated without model loading."""
    print("\nTesting class initialization (without model)...")
    
    try:
        from style_analyzer import StyleAnalyzer
        analyzer = StyleAnalyzer()
        print("✓ StyleAnalyzer initialized")
    except Exception as e:
        print(f"✗ StyleAnalyzer failed: {e}")
        return False
    
    try:
        from semantic_analyzer import SemanticAnalyzer
        analyzer = SemanticAnalyzer()
        print("✓ SemanticAnalyzer initialized")
    except Exception as e:
        print(f"✗ SemanticAnalyzer failed: {e}")
        return False
    
    return True


def test_basic_analysis():
    """Test basic analysis methods without embeddings."""
    print("\nTesting basic analysis methods...")
    
    import numpy as np
    from style_analyzer import StyleAnalyzer
    from semantic_analyzer import SemanticAnalyzer
    
    # Create dummy data
    texts = ["This is a test.", "Another test sentence."]
    embeddings = np.random.randn(2, 768)  # Simulating BERT embeddings
    
    try:
        style_analyzer = StyleAnalyzer()
        style_results = style_analyzer.analyze(texts, embeddings)
        
        assert 'formality' in style_results
        assert 'complexity' in style_results
        assert 'sentiment' in style_results
        print("✓ Style analysis works")
    except Exception as e:
        print(f"✗ Style analysis failed: {e}")
        return False
    
    try:
        semantic_analyzer = SemanticAnalyzer()
        semantic_results = semantic_analyzer.analyze(texts, embeddings)
        
        assert 'similarity_matrix' in semantic_results
        assert 'coherence' in semantic_results
        print("✓ Semantic analysis works")
    except Exception as e:
        print(f"✗ Semantic analysis failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Validation Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_class_initialization():
        tests_passed += 1
    
    if test_basic_analysis():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print(f"\n✗ {tests_total - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
