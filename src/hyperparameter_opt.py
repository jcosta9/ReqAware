import os
import torch
import optuna
import argparse
from pathlib import Path
import copy
from datetime import datetime
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss

# Project-specific imports
from models.architectures import CBMSequentialEfficientNetFCN
from models.loss.custom_fuzzy_loss import CustomFuzzyLoss
from train_cbm import load_config

def objective(trial, base_config, args):
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
        rules[rule_name].fuzzy_lambda = trial.suggest_float(lambda_key, 0.1, 0.99)
        
        # Sample p value for operators if they exist
        if hasattr(rules[rule_name], 'operators'):
            p_value = trial.suggest_float(p_key, 1.0, 4.0)
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
    if not args.debug:
        print(f"Starting training on {config.device}")
        trainer.train()  # This will train the model
    
    # Get predictions and calculate metrics
    concept_logits, concept_predictions, concept_ground_truth, _ = (
        trainer.concept_predictor_trainer.get_predictions(dataloader=val_loader)
    )
    _, validation_acc = trainer.concept_predictor_trainer.test(dataloader=val_loader)
    val_acc_test = accuracy_score(concept_predictions, concept_ground_truth)
    # Calculate fuzzy loss using the original fuzzy loss function
    fuzzy_loss_fn = CustomFuzzyLoss(
        base_config.concept_predictor.fuzzy_loss, 
        current_loss_fn=BCEWithLogitsLoss()
    )
    fuzzy_loss_fn.to(device)

    logit_tensor = torch.tensor(concept_logits)
    y_true = torch.tensor(concept_ground_truth)
    
    with torch.no_grad():
        total_loss = fuzzy_loss_fn(logit_tensor, y_true)
        standard_loss = fuzzy_loss_fn.last_standard_loss.item()
        fuzzy_loss_value = fuzzy_loss_fn.last_fuzzy_loss.item()
    
    # Print trial results
    print(f"Trial {trial.number}:")
    print(f"  Validation Accuracy: {validation_acc:.4f}")
    print(f"  Standard Loss: {standard_loss:.4f}")
    print(f"  Base Fuzzy using default values: {fuzzy_loss_value:.4f}")
    
    # Return metrics as a dictionary
    return validation_acc, standard_loss, fuzzy_loss_value

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=20, 
                        help="Number of trials to run")
    parser.add_argument("--output-dir", type=str, default="./optuna_results",
                        help="Directory to save results")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (skip training)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"cbm_optimization_{timestamp}"
    
    # Load base configuration
    base_config = load_config(Path("files/configs/GTSRB_CBM_config_rules_set1.yaml"))
    
    # Create and configure the study
    # We want to minimize standard_loss and fuzzy rule loss and maximize validation_accuracy
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{os.path.join(args.output_dir, study_name)}.db",
        load_if_exists=True,
        directions=["minimize", "maximize", "minimize"]  # standard_loss, accuracy, fuzzy_loss
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, args), 
        n_trials=args.n_trials,
    )
    
    # Get Pareto front after optimization
    pareto_front = study.best_trials
    
    # Print best hyperparameters
    print("\nBest trials (Pareto front):")
    for i, trial in enumerate(pareto_front):
        print(f"\nTrial {i+1}:")
        print(f"  Validation Accuracy: {trial.values[0]:.4f}")
        print(f"  Validation Standard Loss: {trial.values[1]:.4f}")
        print(f"  Fuzzy Loss: {trial.values[2]:.4f}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
