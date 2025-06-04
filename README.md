# Graph-based Price Attribution

This is the official repository for "Graph-based Price Attribution" (GPA), a novel approach to solving retail price attribution using graph neural networks.

## Getting Started

### Install Project Dependencies

`gpa` is managed via the `uv` package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)). To install the dependencies, simply run `uv sync` from the root directory of the repository after cloning.

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting and run code quality checks. Any issues must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

## Viewing a Dataset

To view a dataset, simply use the [Data Viewer](notebooks/data_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual price graphs.


## Training a Model

To train a model, first fill out a config (using the [example config](src/gpa/training/sample_config.yaml) as a template). Then, run the [training script](src/gpa/training/train_attributor.py):

```bash
uv run train --config path/to/your/config.yaml
```

The training script will save trained weights (both the best in terms of validation loss and the most recent copy) to the checkpoint directory specified in the config, and metrics will be logged in Weights and Biases (if indicated in the config) or locally (to the log directory specified in the config). The train config will also be saved in this log directory.

Use the [Metrics Viewer](notebooks/metrics_viewer.ipynb) to view loss curves and other metrics from a (locally logged) training run.

## Evaluating a Model

Models can be evaluated via the [Evaluation Script](src/gpa/evaluation/evaluate_attributor.py). To run the evaluation script, use the following command:

```bash
uv run eval --ckpt path/to/model/checkpoint.ckpt
```

This script will follow the logging settings specified in the config (WandB vs. local). It will also save evaluation metrics to a YAML file in the log directory.

To get a qualitative sense of how well a model performs for price attribution, use the [Predictions Viewer](notebooks/predictions_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual price attribution predictions (and comparing them to the ground truth).

## Development

### Managing Dependencies

To add a new dependency to the project, run `uv add <package-name>`. This will install the dependency into uv's managed .venv and automatically update the `pyproject.toml` file and the `uv.lock` file, ensuring that the dependency is available for all users of the project who run `uv sync`.

To remove a dependency, run `uv remove <package-name>`. This will perform the reverse of `uv add` (including updating the `pyproject.toml` and `uv.lock` files).

See [uv's documentation](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) for more details.
