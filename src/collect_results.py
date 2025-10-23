import glob
import os, sys
from pathlib import Path

import pandas as pd

from config import load_config
from config.standard_trainer_config import StandardTrainerConfig
from utils.evaluation import evaluate_model_results
from utils.experiment_initialization import load_cbm_model, load_standard_model



# Set the root directory
EXPERIMENTS_DIR = Path('./experiments/RQ/')
OUTPUT_DIR = EXPERIMENTS_DIR / "Results"
REQAWARE_DIR = EXPERIMENTS_DIR / "reqaware"
VANILLA_CBM_DIR = EXPERIMENTS_DIR / "vanilla_cbm"
BASELINE_CNN_DIR = EXPERIMENTS_DIR / "baseline_cnn"

BASE_CONFIG_PATH = Path("../files/configs/")
REQAWARE_DEFAULT_CONFIG_FILE = BASE_CONFIG_PATH / "RQ1_GTSRB_REQAWARE_config.yaml"
VANILLA_CBM_DEFAULT_CONFIG_FILE = BASE_CONFIG_PATH / "RQ1_GTSRB_VANILLA_CBM_config.yaml"
BASELINE_CNN_DEFAULT_CONFIG_FILE = BASE_CONFIG_PATH / "RQ1_GTSRB_BASELINE_CNN_config.yaml"



def get_files(dir:Path):
    model_files = []
    pattern = os.path.join(dir, '**', '*.pt')

    pt_files = glob.glob(pattern, recursive=True)

    for file_path in pt_files:
        model_files.append(file_path)
    
    return model_files

def get_cbm_results(model_names:list, 
                      model_config, 
                      train_dataloader, 
                      val_loader, 
                      test_loader, 
                      nametag:str,
                      output_dir=OUTPUT_DIR,
                      prediction_threshold=0.5):
    concepts_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    i = 0

    for model_name in model_names:
        print(f"iteration {i}")
        print(f"Evaluating Baseline CBM model: {model_name}")
        trainer = load_cbm_model(model_config, 
                                 model_name,
                                 train_dataloader, #
                                 val_loader, 
                                 test_loader)
        trainer.concept_predictor_trainer.prediction_threshold = prediction_threshold
        concept_results = evaluate_model_results(trainer.concept_predictor_trainer, test_loader, seed=model_name.split("_")[2])
        label_results = evaluate_model_results(trainer.label_predictor_trainer, test_loader, seed=model_name.split("_")[2])

        concepts_df = pd.concat([concepts_df, concept_results])
        labels_df = pd.concat([labels_df, label_results])

        i += 1
    
    concepts_df.to_excel(output_dir / f"{nametag}_concept_results.xlsx")
    labels_df.to_excel(output_dir / f"{nametag}_label_results.xlsx")
    return concepts_df, labels_df


def get_cnn_results(model_names:list, 
                      model_config, 
                      train_dataloader, 
                      val_loader, 
                      test_loader, 
                      nametag:str,
                      output_dir=OUTPUT_DIR,
                      prediction_threshold=0.5):
    
    labels_df = pd.DataFrame()
    i = 0
    for model_name in model_names:
        print(f"iteration {i}")
        print(f"Evaluating Baseline CBM model: {model_name}")
        trainer = load_standard_model(model_config, 
                                        model_name,
                                        train_dataloader, #
                                        val_loader, 
                                        test_loader)
        label_results = evaluate_model_results(trainer, test_loader, seed=model_name.split("_")[2])

        # concepts_df = pd.concat([concepts_df, concept_results])
        labels_df = pd.concat([labels_df, label_results])
        i += 1

    labels_df.to_excel(output_dir / f"{nametag}_label_results.xlsx")
    return labels_df

if __name__ == "__main__":
    os.chdir(sys.path[0])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Configs")
    reqaware_config = load_config(REQAWARE_DEFAULT_CONFIG_FILE)
    vcbm_config = load_config(VANILLA_CBM_DEFAULT_CONFIG_FILE)
    bcnn_config = load_config(
        BASELINE_CNN_DEFAULT_CONFIG_FILE,
        configClass=StandardTrainerConfig,
        )


    print("Loading datasets")
    (train_loader,
        val_loader,
        test_loader) = (bcnn_config.dataset
                            .factory(
                                seed=bcnn_config.seed, 
                                config=bcnn_config.dataset
                            )
                            .get_dataloaders()
                        )
    
    (concept_train_loader,
    concept_val_loader,
    concept_test_loader) = (reqaware_config.dataset
                                .factory(
                                    seed=reqaware_config.seed, 
                                    config=reqaware_config.dataset
                                )
                                .get_dataloaders()
                            )

    print('ReqAware')
    print(get_files(REQAWARE_DIR))
    get_cbm_results(
        model_names=get_files(REQAWARE_DIR), 
        model_config=reqaware_config, 
        train_dataloader=concept_train_loader, 
        val_loader=concept_val_loader, 
        test_loader=concept_test_loader,
        nametag='ReqAware',
        output_dir=OUTPUT_DIR,
        prediction_threshold=0.5
        )

    print('Vanilla CBM')
    print(get_files(VANILLA_CBM_DIR))
    get_cbm_results(
        model_names=get_files(VANILLA_CBM_DIR), 
        model_config=vcbm_config, 
        train_dataloader=concept_train_loader, 
        val_loader=concept_val_loader, 
        test_loader=concept_test_loader,
        nametag='VanillaCBM',
        output_dir=OUTPUT_DIR,
        prediction_threshold=0.5
        )
    

    print('Baseline CNN')
    print(get_files(BASELINE_CNN_DIR))
    get_cnn_results(
        model_names=get_files(BASELINE_CNN_DIR), 
        model_config=bcnn_config, 
        train_dataloader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader,
        nametag='BaselineCNN',
        output_dir=OUTPUT_DIR,
        prediction_threshold=0.5
        )