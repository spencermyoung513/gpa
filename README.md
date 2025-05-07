# Graph-based Price Attribution

This is the official repository for "Graph-based Price Attribution" (GPA), a novel approach to solving retail price attribution using graph neural networks.

## Getting Started

### Install Project Dependencies

```bash
conda create --name gpa python=3.10
conda activate gpa
pip install -r requirements.txt
```

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting. There will also be a flake8 check that provides warnings about various Python styling violations. These must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

## Viewing a Dataset

To view a dataset, simply run the [Data Viewer](notebooks/data_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual price graphs.
