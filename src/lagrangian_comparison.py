import os
import time
from pathlib import Path
import yaml
import subprocess
import multiprocessing as mp
from multiprocessing import Process

import torch

from config import load_config, StandardTrainerConfig
from models.architectures import CBMSequentialEfficientNetFCN, EfficientNetv2
from models.trainer import StandardTrainer
from models.trainer.cbm_trainer import CBMTrainer

rho_list = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
lr_list = [0.001, 0.0025, 0.005, 0.0075, 0.01]

gpu_in_use = {}

def try_gpus():
    pass


# RACE CONDITION!!!!
def get_available_gpus():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ).decode()

    free = []

    for line in result.strip().split("\n"):
        idx, mem = line.split(",")
        if int(mem) > 10000:
            free.append(int(idx))

    return free


def assign_gpu():
    for gpu in get_available_gpus():
#
 #       if gpu not in gpu_in_use:
  #          return gpu
#
 #       if not gpu_in_use[gpu].is_alive():
  #          gpu_in_use[gpu].join()
   #         del gpu_in_use[gpu]
            return gpu

    return None



def execute_experiment(experiment_id: int, rho: float, lr: float, gpu: int):

    # RACE CONDITION!!!!
    with open("files/configs/GTSRB_CBM_config_best_trial_loading.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["experiment_id"] = f"id: {experiment_id}, rho: {rho}, lr: {lr}"
    config["device_no"] = gpu

    with open("files/configs/GTSRB_CBM_config_best_trial_loading.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


    config_cbm = load_config(Path("files/configs/GTSRB_CBM_config_best_trial_loading.yaml"))

    config_cbm.concept_predictor.fuzzy_loss.use_fuzzy_loss = True
    config_cbm.concept_predictor.fuzzy_loss.use_lagrangian_optimization = True
    config_cbm.concept_predictor.fuzzy_loss.rho = rho
    config_cbm.concept_predictor.lr = lr
    config_cbm.concept_predictor.gpu = gpu

    dataset_factory = config_cbm.dataset.factory(
        seed=config_cbm.seed, config=config_cbm.dataset
    ).set_dataloaders()

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
    jobs = []

    experiment_id = 0
    for rho in rho_list:
        for lr in lr_list:
            jobs.append((experiment_id, rho, lr))
            # execute_experiment(dataset_factory, experiment_id, rho, lr)
            experiment_id += 1
    
    while jobs or gpu_in_use:

        gpu = assign_gpu()

        # if there is a gpu free and there are still jobs
        if gpu is not None and jobs:

            exp_id, rho, lr = jobs.pop(0)

            p = Process(
                target=execute_experiment,
                args=(exp_id, rho, lr, gpu),
            )

            # mark gpu as in_use
            #gpu_in_use[gpu] = p

            p.start()


            print(f"Started experiment {exp_id} on GPU {gpu}")

            # lower risks of race condtions by using sleep()
            time.sleep(50)



if __name__ == "__main__":
    print("starting main")
    print(get_available_gpus())

    mp.set_start_method("spawn", force=True)
    main()
