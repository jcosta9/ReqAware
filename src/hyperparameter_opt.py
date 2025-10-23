import os
import torch
import optuna
import argparse
from pathlib import Path
import copy
from datetime import datetime
import multiprocessing
from multiprocessing import Pool
from torch.nn import BCEWithLogitsLoss

# Project-specific imports
from models.trainer import StandardTrainer
from models.trainer.cbm_trainer import CBMTrainer
from models.architectures import CBMSequentialEfficientNetFCN, EfficientNetv2
from models.loss.custom_fuzzy_loss import CustomFuzzyLoss
from train_cbm import cbm_load_config
from train import cnn_load_config
    
def objective_baseline_cnn(trial, base_config, args):
    """Objective function for baseline CBM (without fuzzy loss)."""
    # Create a deep copy of the base config to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # Set device
    if not torch.cuda.is_available():
        config.device = 'cpu'
        print("using cpu")
        
    # Sample hyperparameter
    config.training.lr = trial.suggest_float("concept_lr", 0.0001, 0.01, log=True)
    
    # Set up dataset
    dataset_factory = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).set_dataloaders()

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader
    
    model = EfficientNetv2(config.dataset.n_labels).to(config.device)

    # Train
    trainer = StandardTrainer(
        config=config.training,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.device,
    )
    trainer.train()
            
    # Get concept validation metrics
    _, validation_acc = trainer.test(dataloader=val_loader)
    
    torch.cuda.empty_cache()
    
    # Return the metric we want to optimize
    return validation_acc

def objective_baseline_cbm(trial, base_config, args):
    """Objective function for baseline CBM (without fuzzy loss)."""
    # Create a deep copy of the base config to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # Set device
    if not torch.cuda.is_available():
        config.device = 'cpu'
        print("using cpu")
        
    # Sample CBM-specific hyperparameters
    config.concept_predictor.lr = trial.suggest_float("concept_lr", 0.0001, 0.01, log=True)
    config.label_predictor.lr = trial.suggest_float("label_lr", 0.0001, 0.01, log=True)
    
    # Make sure fuzzy loss is disabled
    config.concept_predictor.fuzzy_loss.use_fuzzy_loss = False
    
    # Set up dataset
    dataset_factory = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).set_dataloaders()

    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader
    
    # Initialize model
    model = CBMSequentialEfficientNetFCN(config)
    
    # Initialize trainer
    trainer = CBMTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    # Training with error handling for NaNs
    try:
        if not args.debug:
            trainer.train()
    except RuntimeError as e:
        if "encountered nan or inf" in str(e).lower() or "nan" in str(e).lower():
            print(f"Trial {trial.number} failed due to NaN values: {e}")
            raise optuna.exceptions.TrialPruned()
        else:
            raise
            
    # Get concept validation metrics
    _, validation_acc = trainer.concept_predictor_trainer.test(dataloader=val_loader)
    
    torch.cuda.empty_cache()
    
    # Return the metric we want to optimize
    return validation_acc
    
def fuzzy_objective(trial, base_config, args):
    """Objective function to be optimized by Optuna.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration to modify
        available_gpus: List of available GPU devices
        args: Command-line arguments
        
    Returns:
        Dictionary containing the metrics (loss, accuracy, etc.)
    """
    # Create a deep copy of the base config to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # if cuda is available us the device from the config (where all the data is)
    if not torch.cuda.is_available():
        device = 'cpu'
        config.device = 'cpu'
        print("using cpu")
    else:
        device = f"{config.device}"
    
    # Sample hyperparameters for this trial
    config.concept_predictor.lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    
    # Update lambda and p values for fuzzy logic rules
    rules = config.concept_predictor.fuzzy_loss.rules
    for rule_name in rules:
        lambda_key = f"lambda_{rule_name}"
        p_key = f"p_{rule_name}"
        
        # Sample lambda value for this rule
        rules[rule_name].fuzzy_lambda = trial.suggest_float(lambda_key, 0.0001, 0.8)
        
        # Sample p value for operators if they exist
        if hasattr(rules[rule_name], 'operators'):
            p_value = trial.suggest_float(p_key, 1.01, 30.0)
            for op_name in ['t_norm', 't_conorm', 'e_aggregation', 'a_aggregation']:
                if hasattr(rules[rule_name].operators, op_name):
                    op = getattr(rules[rule_name].operators, op_name)
                    if hasattr(op, 'params') and 'p' in op.params:
                        op.params['p'] = p_value
    
    # Set up dataset
    dataset_factory = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).set_dataloaders()
    
    train_loader = dataset_factory.train_dataloader
    val_loader = dataset_factory.val_dataloader
    test_loader = dataset_factory.test_dataloader
    
    # Initialize model
    model = CBMSequentialEfficientNetFCN(config)
    
    # Initialize trainer
    from models.trainer.cbm_trainer import CBMTrainer
    trainer = CBMTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    # If debug mode, skip actual training to test optimization loop
    try:
        if not args.debug:
            print(f"Starting training on {config.device}")
            trainer.train()  # This will train the model
    except RuntimeError as e:
            if "encountered nan or inf" in str(e).lower() or "nan" in str(e).lower():
                print(f"Trial {trial.number} failed due to NaN values: {e}")
                raise optuna.exceptions.TrialPruned()
            else:
                # Re-raise other runtime errors
                raise
    # Get predictions and calculate metrics
    concept_logits, concept_predictions, concept_ground_truth, _ = (
        trainer.concept_predictor_trainer.get_predictions(dataloader=val_loader)
    )
    concept_predictions_tensor = torch.tensor(concept_predictions, device=device)
    concept_ground_truth_tensor = torch.tensor(concept_ground_truth, device=device)
    all_correct_predictions = torch.sum(torch.all(concept_predictions_tensor == concept_ground_truth_tensor, dim=1)).item()
    total_samples = len(concept_ground_truth)
    per_prediction_accuracy = all_correct_predictions / total_samples

    # Calculate fuzzy loss using the original fuzzy loss function
    fuzzy_loss_fn = CustomFuzzyLoss(
        base_config.concept_predictor.fuzzy_loss, 
        current_loss_fn=BCEWithLogitsLoss()
    )
    fuzzy_loss_fn.to(device)

    logit_tensor = torch.tensor(concept_logits)
    y_true = torch.tensor(concept_ground_truth)
    
    with torch.no_grad():
        _ = fuzzy_loss_fn(logit_tensor, y_true)
        standard_loss = fuzzy_loss_fn.last_standard_loss.item()
        fuzzy_loss_value = fuzzy_loss_fn.last_fuzzy_loss.item()
    
    torch.cuda.empty_cache()
    # Print trial results
    print(f"Trial {trial.number}:")
    print(f"  Validation Accuracy: {per_prediction_accuracy:.4f}")
    print(f"  Standard Loss: {standard_loss:.4f}")  
    print(f"  Base Fuzzy using default values: {fuzzy_loss_value:.4f}")
    
    # Return metrics as a dictionary
    return per_prediction_accuracy

def create_objective_function(model_type):
    # Return the appropriate objective function based on model type
    if model_type == "baseline_cbm":
        return objective_baseline_cbm
    elif model_type == "baseline_cnn":
        return objective_baseline_cnn
    else:
        return fuzzy_objective
    
def run_optimization(gpu_id, study, study_name, base_config, n_trials, args):

    process_id = os.getpid()
    print(f"\n[Worker Info] Process ID: {process_id} | Running on GPU: {gpu_id}")
    print(f"[Worker Info] Will run {n_trials} trials for study: {study_name}")

    base_config.device = f"cuda:{gpu_id}"
    
    # Run optimization
    study.optimize(
        lambda trial: fuzzy_objective(trial, base_config, args), 
        n_trials=n_trials,
    )
    
    # Get Pareto front after optimization
    pareto_front = study.best_trials
    
    return pareto_front

def run_worker(worker_args):
    """Worker function that runs optimization on a specific GPU.
    
    Args:
        worker_args: Tuple containing (gpu_id, study_name, base_config, n_trials, args)
    """
    gpu_id, study, study_name, base_config, n_trials, args = worker_args
    return run_optimization(gpu_id, study, study_name, base_config, n_trials, args)

def run_worker_with_objective(worker_args):
    """Worker function that runs optimization on a specific GPU with custom objective function.
    
    Args:
        worker_args: Tuple containing (gpu_id, study, study_name, base_config, n_trials, args, objective_func)
    """
    gpu_id, study, study_name, base_config, n_trials, args, objective_func = worker_args
    return run_optimization_with_objective(gpu_id, study, study_name, base_config, n_trials, args, objective_func)

def run_optimization_with_objective(gpu_id, study, study_name, base_config, n_trials, args, objective_func):
    process_id = os.getpid()
    print(f"\n[Worker Info] Process ID: {process_id} | Running on GPU: {gpu_id}")
    print(f"[Worker Info] Will run {n_trials} trials for study: {study_name}")

    base_config.device = f"cuda:{gpu_id}"
    
    # Run optimization with the provided objective function
    study.optimize(
        lambda trial: objective_func(trial, base_config, args), 
        n_trials=n_trials,
    )
    
    # Get best trials after optimization
    best_trials = study.best_trials
    
    return best_trials

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=20, 
                        help="Number of trials to run")
    parser.add_argument("--output-dir", type=str, default="./optuna_results",
                        help="Directory to save results")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (skip training)")
    parser.add_argument("--gpu_ids", type=str, default="0", 
                        help="The set of gpu ids to use. Each is assigned a worker that does optimization.")
    parser.add_argument("--study_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), 
                        help="Sets the study name")
    parser.add_argument("--model-type", type=str, choices=["cbm_fuzzy", "baseline_cbm", "baseline_cnn"], 
                        default="cbm_fuzzy", help="Model type to optimize")
    args = parser.parse_args()
    
    # Initialize CUDA before forking processes
    if torch.cuda.is_available():
        torch.cuda.init()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set study name based on model type
    model_type_prefix = f"{args.model_type}_"
    study_name = f"{model_type_prefix}optimization_{args.study_name}"
    
    # Load appropriate base configuration
    if args.model_type == "cbm_fuzzy":
        base_config = cbm_load_config(Path("files/configs/GTSRB_CBM_config_best_trial_loading.yaml"))
    elif args.model_type == "baseline_cbm":
        base_config = cbm_load_config(Path("files/configs/GTSRB_CBM_config.yaml"))
    elif args.model_type == "baseline_cnn":
        base_config = cnn_load_config("files/configs/GTSRB_Baseline_config.yaml")
        
    
    # Get objective function for this model type
    objective_func = create_objective_function(args.model_type)
    
    gpu_ids = [int(id) for id in args.gpu_ids.split(",")]

    # Calculate the number of trials per process
    if gpu_ids:
        n_trial_process = int(args.n_trials / len(gpu_ids))  # Convert to int
    else:
        n_trial_process = args.n_trials

    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{os.path.join(args.output_dir, study_name)}.db",
        load_if_exists=True,
        direction="maximize"  # Assuming higher accuracy is better
    )
    
    # Create worker args
    worker_args = [(gpu_id, study, study_name, base_config, n_trial_process, args, objective_func) 
                   for gpu_id in gpu_ids]
    
    # Run optimization in parallel
    with Pool(processes=len(gpu_ids)) as pool:
        results = pool.map(run_worker_with_objective, worker_args)
        
    # Process results
    print(f"\n======= COMBINED RESULTS FROM ALL WORKERS ({args.model_type}) =======")
    
    # Collect all best trials from all workers
    best_trials = []
    for i, worker_best_trials in enumerate(results):
        gpu_id = gpu_ids[i]
        print(f"\nGPU {gpu_id} found {len(worker_best_trials)} best trials")
        best_trials.extend(worker_best_trials)
    
    print(f"\nCombined best trials count: {len(best_trials)}")
    
    # Sort by validation accuracy (descending)
    sorted_trials = sorted(best_trials, key=lambda t: t.value, reverse=True)
    
    # Print top trials
    print(f"\nTop trials for {args.model_type}:")
    for i, trial in enumerate(sorted_trials[:5]):
        if i >= len(sorted_trials):
            break
        print(f"\nTrial {i+1}:")
        print(f"  Validation Accuracy: {trial.value:.4f}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    multiprocessing.set_start_method(method="spawn")
    main()
