from pathlib import Path
from omegaconf import OmegaConf

from models.architectures import CBMSequentialEfficientNetFCN
from config import CBMTrainerConfig
from models.trainer.cbm_trainer import CBMTrainer

from numpy.random import seed as set_numpy_seed
from torch import manual_seed as set_torch_seed
from torch.cuda import manual_seed_all as set_torch_cuda_seed
from random import seed as set_random_seed
from torch.backends import cudnn


def cbm_load_config(config_path) -> CBMTrainerConfig:
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

def set_reproducibility_seed(seed):
    set_random_seed(seed)
    set_numpy_seed(seed)
    set_torch_seed(seed)
    set_torch_cuda_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def main():
    config = cbm_load_config(Path("files/configs/testing_other_e.yaml"))

    set_reproducibility_seed(config.seed)
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


if __name__ == "__main__":
    main()
