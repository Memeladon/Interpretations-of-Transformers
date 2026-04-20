# Contributing to Interpretations-of-Transformers

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Interpretations-of-Transformers framework.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Interpretations-of-Transformers.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`

## Development Setup

### Installing in Development Mode

```bash
pip install -e .
```

This allows you to make changes to the code and test them immediately without reinstalling.

### Running Tests

```bash
python examples/validate.py
```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

## Making Changes

1. **Write Tests**: Add tests for new functionality
2. **Update Documentation**: Update README.md and docstrings as needed
3. **Follow Existing Patterns**: Match the style of existing code
4. **Keep Changes Focused**: Make small, focused commits

## Commit Messages

Write clear, concise commit messages:
- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Add details in the commit body if needed

Example:
```
Add sentiment analysis visualization

- Create new plot function for sentiment scores
- Update visualization module documentation
- Add example to demo notebook
```

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation to reflect your changes
3. Add examples if you're introducing new features
4. Submit a pull request with a clear description of changes

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How have you tested these changes?

## Checklist
- [ ] Code follows the project style guidelines
- [ ] Documentation has been updated
- [ ] Tests pass
- [ ] Examples have been updated (if applicable)
```

## Areas for Contribution

We welcome contributions in these areas:

### New Features
- Additional style metrics
- More semantic analysis methods
- Support for multilingual models
- Interactive visualization tools
- Attention-based interpretation methods

### Improvements
- Performance optimizations
- Better error handling
- Extended documentation
- More examples and tutorials
- Unit tests

### Documentation
- Tutorial notebooks
- Use case examples
- API documentation
- Video demonstrations

## Questions?

If you have questions about contributing, please:
- Open an issue with the "question" label
- Reach out to the maintainers

## Code of Conduct

Be respectful and constructive in all interactions. We aim to create a welcoming environment for all contributors.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Interpretations-of-Transformers!
