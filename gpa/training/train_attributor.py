import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from gpa.configs import TrainingConfig
from gpa.datamodules import PriceAttributionDataModule
from gpa.models.attributors import LightningPriceAttributor


def train(config: TrainingConfig):
    logger = CSVLogger(
        save_dir=config.log_dir,
        name=config.run_name,
    )
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    version_dir = log_dir.stem
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=config.chkp_dir / config.run_name / version_dir,
        filename="best_loss",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = L.Trainer(
        accelerator=config.accelerator.value,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        logger=logger,
        callbacks=[best_ckpt_callback],
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
        balanced_edge_sampling=config.balanced_edge_sampling,
    )
    datamodule = PriceAttributionDataModule(
        data_dir=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    train(config)
