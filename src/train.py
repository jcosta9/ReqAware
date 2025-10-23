from pathlib import Path
from omegaconf import OmegaConf

from models.architectures import EfficientNetv2
from models.trainer import StandardTrainer
from config import StandardTrainerConfig

from config import load_config

def main():
    config = load_config(
        Path("files/configs/GTSRB_Baseline_config.yaml"),
        configClass=StandardTrainerConfig
        )

    # Dataset
    dataset_factory = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).set_dataloaders()

    if dataset_factory is None:
        return

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader

    # Model
    model = EfficientNetv2(config.dataset.n_labels).to(config.device)

    # Train
    trained_model, test_accuracy = StandardTrainer(
        config=config.training,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.device,
    ).train()

    return dataset_factory, trained_model


if __name__ == "__main__":
    main()
