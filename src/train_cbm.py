from pathlib import Path
from omegaconf import OmegaConf

from models.architectures import CBMSequentialEfficientNetFCN
from config import CBMTrainerConfig
from models.trainer.cbm_trainer import CBMTrainer


def load_config(config_path) -> CBMTrainerConfig:
    """
    Load the configuration from a YAML file.
    """
    # Load YAML
    cfg_yaml = OmegaConf.load(config_path)
    cfg_structured = OmegaConf.structured(CBMTrainerConfig)
    cfg = OmegaConf.merge(cfg_structured, cfg_yaml)
    cfg = OmegaConf.to_object(cfg)
    cfg.resolve()
    return cfg


def main():
    config = load_config(Path("files/configs/GTSRB_CBM_config_optimalV3.yaml"))

    # Dataset
    dataset_factory = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).set_dataloaders()

    if dataset_factory is None:
        return

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader

    # # Model
    model = CBMSequentialEfficientNetFCN(config)

    # # Train
    trainer = CBMTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    trainer.train()

    # return dataset_factory, model, trainer


if __name__ == "__main__":
    main()
