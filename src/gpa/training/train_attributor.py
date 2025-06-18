import argparse
from pathlib import Path

import lightning as L
import yaml
from gpa.configs import TrainingConfig
from gpa.datamodules import PriceAttributionDataModule
from gpa.models.attributors import LightningPriceAttributor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger


def train(config: TrainingConfig):
    if config.logging.use_wandb:
        logger = WandbLogger(
            project=config.logging.project_name,
            name=config.run_name,
            tags=["TRAIN"],
        )
    else:
        logger = CSVLogger(
            save_dir=config.logging.log_dir,
            name=config.run_name,
            version="",
        )
    log_dir = config.logging.log_dir / config.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=config.logging.chkp_dir / config.run_name,
        filename="best_loss",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=config.logging.chkp_dir / config.run_name,
        filename="last",
        save_last=True,
        enable_version_counter=False,
    )
    trainer = L.Trainer(
        accelerator=config.accelerator.value,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        logger=logger,
        callbacks=[best_ckpt_callback, last_ckpt_callback],
        enable_model_summary=False,
    )
    model = LightningPriceAttributor(
        encoder_type=config.model.encoder_type,
        encoder_settings=config.model.encoder_settings,
        link_predictor_type=config.model.link_predictor_type,
        link_predictor_settings=config.model.link_predictor_settings,
        num_epochs=config.num_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        gamma=config.gamma,
        balanced_edge_sampling=config.balanced_edge_sampling,
    )
    datamodule = PriceAttributionDataModule(
        data_dir=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_visual_info=config.model.use_visual_info,
        use_spatially_invariant_coords=config.model.use_spatially_invariant_coords,
        initial_connection_strategy=config.model.initial_connection_strategy,
    )
    trainer.fit(model, datamodule=datamodule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    train(config)


if __name__ == "__main__":
    main()
