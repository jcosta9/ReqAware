import ray
import torch
import os
from ray import tune
from pathlib import Path
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from models.architectures import CBMSequentialEfficientNetFCN

from train_cbm import load_config

def objective(config):
    """Objective function that will be optimized by ray tune. In Ray this constitutes one trial run with some hyperparameters."""
    
    base_config = load_config(Path("../files/configs/GTSRB_CBM_config.yaml"))

    # ---- setting the gpu for the trial run.
    if torch.cuda.is_available():
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            device_no = 0
        else:
            device_no = 0  # Default if no specific GPU is assigned
    else:
        device_no = -1  # Use CPU if CUDA is not available
    if torch.cuda.is_available():
        base_config.device = "cuda"
        base_config.device_no = device_no
    else:
        base_config.device = "cpu"
        base_config.device_no = -1

    # ---- updating the hyperparameters
    rules = base_config.concept_predictor.fuzzy_loss.rules
    for rule_name in rules:
        print(rule_name)

    # ---- setting up the training with the new config
    dataset_factory = base_config.dataset.factory(
        seed=base_config.seed, config=base_config.dataset
    ).set_dataloaders()
    
    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader
    
    model = CBMSequentialEfficientNetFCN(base_config)
    
    from models.trainer.cbm_trainer import CBMTrainer
    trainer = CBMTrainer(
        config=base_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    #trainer.train()
    print(f"Starting training on device detected by Ray: {base_config.device, base_config.device_no}")


def main():
    ray.init(num_gpus=torch.cuda.device_count())
    print(f"Available resources: {ray.available_resources()}")

    base_config = load_config(Path("../files/configs/GTSRB_CBM_config.yaml"))
    objective(base_config)
    

if __name__ == "__main__":
    main()
