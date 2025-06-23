import argparse
from pathlib import Path
from pprint import pprint

import lightning as L
import yaml
from gpa.configs import EvalConfig
from gpa.configs import TrainingConfig
from gpa.datamodules import PriceAttributionDataModule
from gpa.models.attributors import LightningPriceAttributor

import wandb


def evaluate(config: EvalConfig):
    with open(config.trn_config_path, "r") as f:
        trn_config = TrainingConfig(**yaml.safe_load(f))

    evaluator = L.Trainer(
        accelerator=trn_config.accelerator.value,
        enable_model_summary=False,
        logger=False,
    )
    model = LightningPriceAttributor.load_from_checkpoint(config.chkp_path)
    datamodule = PriceAttributionDataModule(
        data_dir=trn_config.dataset_dir,
        batch_size=trn_config.batch_size,
        num_workers=trn_config.num_workers,
        use_visual_info=trn_config.model.use_visual_info,
        use_spatially_invariant_coords=trn_config.model.use_spatially_invariant_coords,
        initial_connection_config=trn_config.model.initial_connection,
    )
    datamodule.setup("")

    if trn_config.logging.use_wandb:
        wandb.init(
            project=trn_config.logging.project_name,
            name=trn_config.run_name,
            tags=["EVAL"],
        )

    metrics = evaluator.test(
        model=model,
        datamodule=datamodule,
        verbose=False,
    )[0]
    config.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.results_dir / "eval_metrics.yaml"
    with open(results_path, "w") as f:
        yaml.dump(metrics, f)

    if trn_config.logging.use_wandb:
        wandb.log(metrics)
        wandb.finish()
    else:
        pprint(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        eval_config = EvalConfig(**yaml.safe_load(f))
    evaluate(eval_config)


if __name__ == "__main__":
    main()
