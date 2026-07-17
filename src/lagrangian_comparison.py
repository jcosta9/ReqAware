import os
from pathlib import Path
from datetime import datetime
import yaml
import subprocess
from multiprocessing import Process

running = {}

import torch

from config import load_config, StandardTrainerConfig
from models.architectures import CBMSequentialEfficientNetFCN, EfficientNetv2
from models.trainer import StandardTrainer
from models.trainer.cbm_trainer import CBMTrainer

rho_list = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
lr_list = [0.001, 0.0025, 0.005, 0.0075, 0.01]

gpu_in_use = []

def try_gpus():
    pass

def get_available_gpus():
    max_memory_mb = 1000
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).decode()

    free = []

    for line in result.strip().split("\n"):
        idx, mem = line.split(",")
        if int(mem) < max_memory_mb:
            free.append(int(idx))

    return free

def assign_gpu():
    pass



def execute_experiment(dataset_factory, experiment_id: int, rho: float, lr:float):

    with open("files/configs/GTSRB_CBM_config_best_trial_loading.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["experiment_id"] = f"id: {experiment_id}, rho: {rho}, lr: {lr}"

    with open("files/configs/GTSRB_CBM_config_best_trial_loading.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


    config_cbm = load_config(Path("files/configs/GTSRB_CBM_config_best_trial_loading.yaml"))

    config_cbm.concept_predictor.fuzzy_loss.use_fuzzy_loss = True
    config_cbm.concept_predictor.fuzzy_loss.use_lagrangian_optimization = True
    config_cbm.concept_predictor.fuzzy_loss.rho = rho
    config_cbm.concept_predictor.lr = lr

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader

    model = CBMSequentialEfficientNetFCN(config_cbm)
    
    trainer = CBMTrainer(
        config=config_cbm,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    trainer.train()

    return
   


def main():
    config_cbm = load_config(Path("files/configs/GTSRB_CBM_config_best_trial_loading.yaml"))

    dataset_factory = config_cbm.dataset.factory(
        seed=config_cbm.seed, config=config_cbm.dataset
    ).set_dataloaders()

    if dataset_factory is None:
        return


    experiment_id = 0
    for rho in rho_list:
        for lr in lr_list:
            execute_experiment(dataset_factory, experiment_id, rho, lr)


if __name__ == "__main__":
    # main()
    print(get_available_gpus())