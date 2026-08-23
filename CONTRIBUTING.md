# Contributing to RankFuse

Thank you for your interest in contributing to RankFuse! We welcome all contributions, including bug fixes, new features, documentation improvements, and bug reports.

## Local Development Setup

To set up a local development environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GauravFrr/RankFuse.git
   cd RankFuse
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies in editable mode**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev,cross-encoder]"
   ```

## Running Tests and Linting

Before submitting a Pull Request, please ensure all checks pass:

1. **Run the test suite**:
   ```bash
   pytest
   ```

2. **Run lint and style checks**:
   ```bash
   ruff check .
   ```

3. **Check formatting**:
   ```bash
   black --check .
   ```

## Submission Guidelines

1. **Create a branch** for your work from `main`.
2. **Make clear, focused commits** with descriptive commit messages.
3. **Open a Pull Request** against the `main` branch. Provide a clear description of the problem your PR solves and how it was tested.
