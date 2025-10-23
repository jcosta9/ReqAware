import argparse
import os
import sys
from pathlib import Path

BASE_CONFIG_PATH = Path("../files/configs/")
REQAWARE_DEFAULT_CONFIG_FILE = BASE_CONFIG_PATH / "RQ1_GTSRB_REQAWARE_config.yaml"
VANILLA_CBM_DEFAULT_CONFIG_FILE = BASE_CONFIG_PATH / "RQ1_GTSRB_VANILLA_CBM_config.yaml"
BASELINE_CNN_DEFAULT_CONFIG_FILE = BASE_CONFIG_PATH / "RQ1_GTSRB_BASELINE_CNN_config.yaml"

from utils import initialize_experiment


if __name__ == "__main__":
    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs='+', required=False, default=42)
    parser.add_argument("--device_no", type=int, required=True)
    parser.add_argument("--baseline_cnn", action='store_true')
    parser.add_argument("--vanilla_cbm", action='store_true')
    parser.add_argument("--reqaware", action='store_true')
    arguments = parser.parse_args()

    print(f"seeds: {arguments.seeds}")
    print(f"Baseline CNN: {arguments.baseline_cnn}")
    print(sys.path[0])
    print(f"found file {BASE_CONFIG_PATH}: {BASE_CONFIG_PATH.exists()}")


    for random_seed in arguments.seeds:
        try:
            random_seed = int(random_seed)
            print(f"Running experiment for seed: {random_seed}")

            if arguments.reqaware:
                print("Training ReqAware CBM")
                train_loader, val_loader, test_loader, model, fuzzy_trainer, config = (
                    initialize_experiment(
                        seed=random_seed, 
                        device_no=arguments.device_no, 
                        config_file=REQAWARE_DEFAULT_CONFIG_FILE,
                        model_type="reqaware"
                    )
                )
                fuzzy_trainer.train()
            
            if arguments.baseline_cnn:
                print("Training Baseline CNN")
                train_loader, val_loader, test_loader, model, standard_trainer, config = (
                    initialize_experiment(
                        seed=random_seed, 
                        device_no=arguments.device_no, 
                        config_file=BASELINE_CNN_DEFAULT_CONFIG_FILE,
                        model_type="baseline_cnn"
                    )
                )
                standard_trainer.train()

            if arguments.vanilla_cbm:
                print("Training Vanilla CBM")
                train_loader, val_loader, test_loader, model, baseline_trainer, config = (
                    initialize_experiment(
                        seed=random_seed, 
                        device_no=arguments.device_no, 
                        config_file=VANILLA_CBM_DEFAULT_CONFIG_FILE,
                        model_type="vanilla_cbm"
                    )
                )
                baseline_trainer.train()

        except Exception as e:
            print(f"Run failed with error: {e}")
    
    print(f"Finished Running the Script!")