# Contributing to AI Test Automation Platform

We welcome contributions to the AI Test Automation Platform! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Add tests for your changes
6. Ensure all tests pass
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)
- API keys for external services (OpenAI, Datadog, Sentry, etc.)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-test-automation-platform.git
cd ai-test-automation-platform
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

5. Initialize the system:
```bash
python main.py init
```

### Development Tools

Install development dependencies:
```bash
pip install -r requirements-dev.txt  # If available
```

Recommended tools:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `pytest` for testing

## Contributing Process

### 1. Issue Creation

Before starting work on a new feature or bugfix:
1. Check existing issues to avoid duplicates
2. Create a new issue with a clear description
3. Wait for maintainer approval before starting work

### 2. Branch Naming

Use descriptive branch names:
- `feature/description` for new features
- `bugfix/description` for bug fixes
- `hotfix/description` for critical fixes
- `docs/description` for documentation updates
- `refactor/description` for code refactoring

### 3. Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```
feat(test-generation): add support for Cypress framework
fix(healing): resolve vision-based element detection issue
docs(api): update endpoint documentation
```

## Coding Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Keep functions small and focused
- Use meaningful variable and function names
- Add docstrings for all public functions and classes

### Code Formatting

Use `black` for consistent formatting:
```bash
black src/
```

### Linting

Use `flake8` for code quality:
```bash
flake8 src/
```

### Type Checking

Use `mypy` for type checking:
```bash
mypy src/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_test_generator.py

# Run with coverage
pytest --cov=src

# Run integration tests
pytest tests/integration/
```

### Writing Tests

- Write unit tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies

Example test structure:
```python
import pytest
from src.test_generation.test_generator import TestCaseGenerator

class TestTestCaseGenerator:
    def test_generate_scenarios_success(self):
        """Test successful scenario generation."""
        generator = TestCaseGenerator()
        scenarios = generator.generate_test_scenarios(
            query="test query",
            max_scenarios=3
        )
        assert len(scenarios) <= 3
        assert all(scenario.title for scenario in scenarios)
    
    def test_generate_scenarios_invalid_query(self):
        """Test scenario generation with invalid query."""
        generator = TestCaseGenerator()
        scenarios = generator.generate_test_scenarios(
            query="",
            max_scenarios=3
        )
        assert len(scenarios) == 0
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type information in docstrings
- Provide usage examples where helpful

### API Documentation

- Update API documentation for new endpoints
- Include request/response examples
- Document error codes and responses

### README Updates

- Update README.md for significant changes
- Include setup instructions for new features
- Update dependency requirements

## Issue Reporting

### Bug Reports

When reporting bugs, include:
1. Clear description of the issue
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, Python version, etc.)
5. Relevant logs or error messages
6. Screenshots if applicable

### Feature Requests

For feature requests, include:
1. Clear description of the feature
2. Use cases and benefits
3. Proposed implementation approach
4. Any relevant examples or references

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Update documentation if needed
3. Add changelog entry
4. Ensure code follows style guidelines
5. Rebase on latest main branch

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #issue_number
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Maintainer approval for merge

## Development Guidelines

### Architecture Principles

- Follow SOLID principles
- Use dependency injection
- Keep components loosely coupled
- Implement proper error handling
- Use async/await for I/O operations

### Security Considerations

- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Implement proper authentication/authorization
- Follow security best practices

### Performance Considerations

- Optimize database queries
- Use caching where appropriate
- Implement proper logging
- Monitor resource usage
- Profile performance-critical code

## Getting Help

- Check existing documentation
- Search existing issues
- Ask questions in discussions
- Contact maintainers for urgent issues

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the AI Test Automation Platform!
