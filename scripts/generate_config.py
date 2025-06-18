"""Randomly form a config file for a model training run.

This script will randomly sample from a number of binary indicators that specify hyperparameters we want to test the effect of.
It will then generate a config with the implied settings, where the config name is a ternary string indicating the status of each sampled indicator.

The ternary will be formatted (0 | 1)(0 | 1)(0 | 1)(0 | 1 | 2 | 3)(0 | 1)(0 | 1).yaml, where:

- The first digit indicates whether/not the model uses a larger MLP for its link predictor (vs. a smaller one)
- The second digit indicates whether/not the model encodes graph nodes using a GNN before performing link prediction
- The third digit indicates whether/not the model uses a translationally-invariant representation of bbox centroids as its input
- The fourth digit indicates which "initial connection heuristic" is used to seed the graph (further processed by a GNN)
- The fifth digit indicates whether/not the model balances the # of positive/negative edges it propagates loss on during each training iteration
- The sixth digit indicates whether/not the model trains with focal loss (gamma = 2.0) vs. regular binary cross-entropy.

All other training settings are shared across model runs (number of epochs, batch size, dataset, etc.)
"""
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import TypeVar

import yaml
from gpa.common.enums import ConnectionStrategy
from gpa.common.enums import EncoderType
from gpa.common.enums import LinkPredictorType
from gpa.configs import LoggingConfig
from gpa.configs import ModelConfig
from gpa.configs import TrainingConfig
from tqdm import tqdm


T = TypeVar("T")


def get_choice(choices: list[T]) -> tuple[T, int]:
    idx = random.randint(0, len(choices) - 1)
    choice = choices[idx]
    return choice, idx


def generate_random_config(save_dir: Path) -> bool:
    layer_widths, layer_widths_idx = get_choice([[128, 64], [128, 128, 64, 32]])
    encoder_type, encoder_idx = get_choice(
        [EncoderType.IDENTITY, EncoderType.TRANSFORMER]
    )
    use_spatially_invariant_coords, spatial_idx = get_choice([False, True])
    initial_connection_strategy, heuristic_idx = (
        get_choice(
            [
                None,
                ConnectionStrategy.NEAREST,
                ConnectionStrategy.NEAREST_BELOW,
                ConnectionStrategy.NEAREST_BELOW_PER_GROUP,
            ]
        )
        if encoder_type != EncoderType.IDENTITY
        else (None, 0)
    )
    balanced_edge_sampling, edge_sampling_idx = get_choice([False, True])
    gamma, focal_loss_idx = get_choice([0.0, 2.0])

    run_name = f"{layer_widths_idx}{encoder_idx}{spatial_idx}{heuristic_idx}{edge_sampling_idx}{focal_loss_idx}"
    save_path = save_dir / f"{run_name}.yaml"
    if save_path.exists():
        return False

    if encoder_type == EncoderType.TRANSFORMER:
        encoder_settings = {"node_hidden_dim": 128, "num_layers": 2}
    else:
        encoder_settings = {}
    link_predictor_settings = {
        "strategy": "concat" if encoder_type == EncoderType.IDENTITY else "hadamard",
        "layer_widths": layer_widths,
        "pi": 0.5,
    }

    model_config = ModelConfig(
        use_visual_info=False,
        use_spatially_invariant_coords=use_spatially_invariant_coords,
        initial_connection_strategy=initial_connection_strategy,
        encoder_type=encoder_type,
        encoder_settings=encoder_settings,
        link_predictor_type=LinkPredictorType.MLP,
        link_predictor_settings=link_predictor_settings,
    )
    logging_config = LoggingConfig(
        use_wandb=True,
        project_name="price-attribution",
    )
    training_config = TrainingConfig(
        run_name=run_name,
        model=model_config,
        logging=logging_config,
        num_epochs=20,
        batch_size=8,
        gamma=gamma,
        balanced_edge_sampling=balanced_edge_sampling,
        dataset_dir=Path("data/price-graphs-ii"),
    )
    with open(save_dir / f"{run_name}.yaml", "w") as f:
        yaml.safe_dump(training_config.model_dump(mode="json"), f)
    return True


def main(n: int, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    num_left = n
    loop = tqdm(desc="Generating configs...", total=n, leave=False)
    while num_left > 0:
        success = generate_random_config(save_dir)
        if success:
            num_left -= 1
            loop.update()
    loop.close()
    print(f"Saved {n} random config(s) to {str(save_dir)}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-n", type=int, default=1, help="The number of configs to generate."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(
            "configs/hparam-search", help="Root directory where configs should be saved"
        ),
    )
    args = parser.parse_args()
    main(n=args.n, save_dir=args.save_dir)
