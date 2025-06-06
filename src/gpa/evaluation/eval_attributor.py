import argparse
from pathlib import Path
from pprint import pprint

import lightning as L
import yaml
from gpa.configs import TrainingConfig
from gpa.datamodules import PriceAttributionDataModule
from gpa.models.attributors import LightningPriceAttributor

import wandb


def evaluate(ckpt_path: Path):
    config_path = Path(str(ckpt_path).replace("chkp", "logs")).with_name("config.yaml")
    with open(config_path, "r") as f:
        config = TrainingConfig(**yaml.safe_load(f))
    trainer = L.Trainer(
        accelerator=config.accelerator.value,
        enable_model_summary=False,
        logger=False,
    )
    model = LightningPriceAttributor.load_from_checkpoint(ckpt_path)
    datamodule = PriceAttributionDataModule(
        data_dir=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_visual_info=config.model.use_visual_info,
        aggregate_by_upc=config.model.aggregate_by_upc,
        use_spatially_invariant_coords=config.model.use_spatially_invariant_coords,
        initial_connection_strategy=config.model.initial_connection_strategy,
    )
    datamodule.setup("")

    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.project_name,
            name=config.run_name,
            tags=["EVAL"],
        )

    metrics = trainer.test(
        model=model,
        datamodule=datamodule,
        verbose=False,
    )[0]
    results_path = config_path.parent / "eval_metrics.yaml"
    with open(results_path, "w") as f:
        yaml.dump(metrics, f)

    if config.logging.use_wandb:
        wandb.log(metrics)
        wandb.finish()
    else:
        pprint(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path)
    args = parser.parse_args()
    evaluate(args.ckpt)


if __name__ == "__main__":
    main()
