import ray
import argparse
import torch
from ray import tune
from pathlib import Path
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from models.architectures import CBMSequentialEfficientNetFCN
from typing import Dict
import copy
from sklearn.metrics import accuracy_score
from models.loss.custom_fuzzy_loss import CustomFuzzyLoss
import json
from datetime import datetime

from train_cbm import load_config

def objective(config: Dict[str, float]):
    """Objective function that will be optimized by ray tune. In Ray this constitutes one trial run with some hyperparameters."""
    
    # creating a copy of the base config to evaluate the fuzzy loss
    base_config = load_config(Path("../files/configs/GTSRB_CBM_config.yaml"))
    original_config = copy.deepcopy(base_config)

    # ---- setting the gpu for the trial run. We assume that each trial gets only one gpu/cpu allocated
    if torch.cuda.is_available():
        base_config.device = "cuda"
        base_config.device_no = 0
    else:
        base_config.device = "cpu"
        base_config.device_no = -1
    
    # ---- updating the hyperparameters, lambdas and p-values for each of the rules
    base_config.concept_predictor.lr = config["lr"]
    rules = base_config.concept_predictor.fuzzy_loss.rules
    for rule_name in rules:
        # Update lambda values if specified
        lambda_key = f"lambda_{rule_name}"
        if lambda_key in config:
            rules[rule_name].fuzzy_lambda = config[lambda_key]
        # updating the p-values
        p_key = f"p_{rule_name}"
        if p_key in config and hasattr(rules[rule_name], 'operators'):
            for op_name in ['t_norm', 't_conorm', 'e_aggregation', 'a_aggregation']:
                if hasattr(rules[rule_name].operators, op_name):
                    op = getattr(rules[rule_name].operators, op_name)
                    if hasattr(op, 'params') and 'p' in op.params:
                        op.params['p'] = config[p_key]

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

    # ---- calculating the trial metrics for tune
    concept_logits, concept_predictions, concept_ground_truth, _ = trainer.concept_predictor_trainer.get_predictions(dataloader=val_loader)
    validation_set_acc = accuracy_score(concept_predictions, concept_ground_truth)
    # constructing a fuzzy loss that is based on the original config
    base_fuzzy_loss = CustomFuzzyLoss(original_config.fuzzy_loss, current_loss_fn=trainer.concept_predictor_trainer.criterion)
    logit_tensor = torch.tensor(concept_logits)

    with torch.no_grad():
        total_loss = base_fuzzy_loss(logit_tensor)
        standard_loss = base_fuzzy_loss.last_standard_loss.item()
        fuzzy_loss_value = base_fuzzy_loss.last_fuzzy_loss.item()
        
        # Get individual rule losses for analysis
        individual_rule_losses = {
            rule_name: loss.item() 
            for rule_name, loss in base_fuzzy_loss.last_individual_losses.items()
        }

    tune.report({"validation_standard_loss": standard_loss, "validation_accuracy": validation_set_acc, "fuzzy_loss": fuzzy_loss_value})

def main():
    
    ray.init(num_gpus=torch.cuda.device_count())
    print(f"Available resources: {ray.available_resources()}")

    # this is the base config that will continuously will be changed by Ray
    base_config = load_config(Path("../files/configs/GTSRB_CBM_config.yaml"))
    
    # ---- this gets all the rules and assigns the lambdas and the p values names
    rule_names = list(base_config.concept_predictor.fuzzy_loss.rules.keys())
    search_space = {}
    for rule in rule_names:
        search_space["lr"] = tune.uniform(0.0001, 0.01)
        search_space[f"lambda_{rule}"] = tune.uniform(0.1, 0.99)
        search_space[f"p_{rule}"] = tune.uniform(1.0, 4.0)
    
    search_strategy = OptunaSearch(metric=["validation_standard_loss","validation_accuracy","fuzzy_loss"],
                                   mode=["min","max","min"])
    
    scheduler = ASHAScheduler(metric="validation_standard_loss", mode="min", max_t=20)
    
    tune_config = tune.TuneConfig(
        search_alg=search_strategy,
        scheduler=scheduler,
        num_samples=50,  # Total number of trials to run
        max_concurrent_trials=torch.cuda.device_count(),  # Run trials equal to number of GPUs
    )

    

if __name__ == "__main__":
    main()
