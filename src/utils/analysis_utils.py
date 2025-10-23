import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Subset

def get_dataset_predictions(model, dataloader, device, dataset_name="dataset", concept_pred_threshold=0.5):
    """
    Generic function to get predictions for any dataset.
    Works with both datasets that have concepts and those that don't.
    """
    model.eval()
    
    all_logits = []
    concept_predictions = []
    concept_probabilities = []
    all_labels = []
    all_concepts = []
    has_concepts = False
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Getting {dataset_name} predictions"):
            # Handle different dataloader formats
            if len(batch_data) == 3:
                idx, inputs, labels_or_tuple = batch_data
                
                # Check if it's a tuple of (concepts, labels)
                if isinstance(labels_or_tuple, (tuple, list)) and len(labels_or_tuple) == 2:
                    concepts, labels = labels_or_tuple
                    has_concepts = True
                    all_concepts.append(concepts.cpu().numpy() if isinstance(concepts, torch.Tensor) else concepts)
                else:
                    labels = labels_or_tuple
            elif len(batch_data) == 2:
                inputs, labels = batch_data
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            inputs = inputs.to(device)
            
            # Forward pass through concept predictor
            outputs = model.concept_predictor(inputs)
            
            # Get probabilities and binary predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > concept_pred_threshold).float()

            
            # Store results
            all_logits.append(outputs.cpu().numpy())
            concept_predictions.append(preds.cpu().numpy())
            concept_probabilities.append(probs.cpu().numpy())
            
            # Handle labels - ensure they're flattened if needed
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.array(labels)
            

            # Flatten if it's a 2D array with single column
            if labels_np.ndim > 1 and labels_np.shape[1] == 1:
                labels_np = labels_np.flatten()
            
            all_labels.append(labels_np)
    
    # Concatenate all batches
    result = {
        'logits': np.vstack(all_logits),
        'predictions': np.vstack(concept_predictions),
        'probabilities': np.vstack(concept_probabilities),
        'labels': np.concatenate(all_labels) if all_labels[0].ndim == 1 else np.vstack(all_labels)
    }
    
    if has_concepts:
        result['concepts'] = np.vstack(all_concepts)
    
    return result

def get_dataset_predictions_cbm(model, dataloader, device, dataset_name="dataset", concept_pred_threshold=0.5):
    """
    Generic function to get predictions for any dataset.
    Works with both datasets that have concepts and those that don't.
    """
    model.eval()
    
    all_logits = []
    concept_predictions = []
    concept_probabilities = []
    all_labels = []
    all_concepts = []
    has_concepts = False
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Getting {dataset_name} predictions"):
            idx, inputs, (concepts, labels) = batch_data
            has_concepts = True
            all_concepts.append(concepts.cpu().numpy() if isinstance(concepts, torch.Tensor) else concepts)
            
            inputs = inputs.to(device)
            
            # Forward pass through concept predictor
            outputs = model.concept_predictor(inputs)
            # print("concepts predicted")
            
            # Get probabilities and binary predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > concept_pred_threshold).float()

            # print("computed!")
            
            # Store results
            all_logits.append(outputs.cpu().numpy())
            concept_predictions.append(preds.cpu().numpy())
            concept_probabilities.append(probs.cpu().numpy())
            # print("results stored")
            
            # Handle labels - ensure they're flattened if needed
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.array(labels)
            
            # print("labels handled")

            # Flatten if it's a 2D array with single column
            if labels_np.ndim > 1 and labels_np.shape[1] == 1:
                labels_np = labels_np.flatten()
            
            all_labels.append(labels_np)
            # print("all labels appended")
    
    # Concatenate all batches
    result = {
        'logits': np.vstack(all_logits),
        'predictions': np.vstack(concept_predictions),
        'probabilities': np.vstack(concept_probabilities),
        'labels': np.concatenate(all_labels) if all_labels[0].ndim == 1 else np.vstack(all_labels)
    }
    
    if has_concepts:
        result['concepts'] = np.vstack(all_concepts)
    
    return result

def analyze_fuzzy_loss_single_model(logits, predictions, fuzzy_loss_fn, model_name, dataset_name):
    """Analyze fuzzy loss for a single model on a single dataset."""
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    
    # Calculate loss
    _ = fuzzy_loss_fn(logits_tensor, predictions_tensor)
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'standard_loss': fuzzy_loss_fn.last_standard_loss.item(),
        'fuzzy_loss': fuzzy_loss_fn.last_fuzzy_loss.item(),
        'total_loss': fuzzy_loss_fn.last_standard_loss.item() + fuzzy_loss_fn.last_fuzzy_loss.item(),
        'rule_losses': {name: loss.item() for name, loss in fuzzy_loss_fn.last_individual_losses.items()}
    }


def compare_fuzzy_losses(baseline_metrics, fuzzy_metrics):
    """Compare fuzzy loss metrics between two models."""
    reduction = baseline_metrics['fuzzy_loss'] - fuzzy_metrics['fuzzy_loss']
    reduction_pct = (reduction / baseline_metrics['fuzzy_loss'] * 100) if baseline_metrics['fuzzy_loss'] > 0 else 0
    
    # Per-rule comparison
    rule_comparison = []
    for rule_name in baseline_metrics['rule_losses'].keys():
        baseline_loss = baseline_metrics['rule_losses'][rule_name]
        fuzzy_loss = fuzzy_metrics['rule_losses'][rule_name]
        improvement = baseline_loss - fuzzy_loss
        improvement_pct = (improvement / baseline_loss * 100) if baseline_loss > 0 else 0
        
        rule_comparison.append({
            'Rule': rule_name,
            'Baseline Loss': baseline_loss,
            'Fuzzy CBM Loss': fuzzy_loss,
            'Improvement': improvement,
            'Improvement %': improvement_pct
        })
    
    return {
        'absolute_reduction': reduction,
        'relative_reduction_pct': reduction_pct,
        'rule_comparison': pd.DataFrame(rule_comparison).sort_values('Improvement %', ascending=False)
    }


def analyze_rule_violations(concept_predictions, dataset_name, model_name, rule_checker):
    """Analyze rule violations for a given set of concept predictions."""
    from collections import defaultdict
    
    total_violations = 0
    flagged_indices = []
    violated_constraints = []
    constraint_counts = defaultdict(int)
    
    for i, pred in enumerate(concept_predictions):
        violations = rule_checker.check_concept_vector(pred.astype(int), verbose=False, early_stop=False)
        if violations:
            total_violations += 1
            flagged_indices.append(i)
            violated_constraints.append(violations)
            
            for violation in violations:
                constraint_counts[violation['constraint']] += 1
    
    violation_rate = (total_violations / len(concept_predictions)) * 100
    
    return {
        'dataset': dataset_name,
        'model': model_name,
        'total_samples': len(concept_predictions),
        'total_violations': total_violations,
        'violation_rate': violation_rate,
        'flagged_indices': flagged_indices,
        'violated_constraints': violated_constraints,
        'constraint_counts': dict(constraint_counts)
    }


def compare_violations(baseline_violations, fuzzy_violations):
    """Compare violation metrics between two models."""
    improvement = baseline_violations['violation_rate'] - fuzzy_violations['violation_rate']
    rel_improvement = (improvement / baseline_violations['violation_rate'] * 100) if baseline_violations['violation_rate'] > 0 else 0
    
    # Per-constraint comparison
    all_constraints = set(baseline_violations['constraint_counts'].keys()) | \
                     set(fuzzy_violations['constraint_counts'].keys())
    
    constraint_comparison = []
    for constraint in sorted(all_constraints):
        baseline_count = baseline_violations['constraint_counts'].get(constraint, 0)
        fuzzy_count = fuzzy_violations['constraint_counts'].get(constraint, 0)
        imp = baseline_count - fuzzy_count
        imp_pct = (imp / baseline_count * 100) if baseline_count > 0 else 0
        
        constraint_comparison.append({
            'Constraint': constraint,
            'Baseline Count': baseline_count,
            'Fuzzy CBM Count': fuzzy_count,
            'Improvement': imp,
            'Improvement %': imp_pct
        })
    
    return {
        'absolute_improvement': improvement,
        'relative_improvement_pct': rel_improvement,
        'constraint_comparison': pd.DataFrame(constraint_comparison).sort_values('Improvement', ascending=False)
    }


def print_fuzzy_loss_results(metrics, comparison=None):
    """Pretty print fuzzy loss results."""
    print(f"\n{metrics['model']} on {metrics['dataset']}:")
    print(f"  Standard BCE Loss:    {metrics['standard_loss']:.10f}")
    print(f"  Fuzzy Rules Loss:     {metrics['fuzzy_loss']:.10f}")
    print(f"  Total Loss:           {metrics['total_loss']:.10f}")
    
    if comparison:
        print(f"\n  Improvement over Baseline:")
        print(f"    Absolute: {comparison['absolute_reduction']:.10f}")
        print(f"    Relative: {comparison['relative_reduction_pct']:.2f}%")


def print_violation_results(violations, comparison=None):
    """Pretty print violation results."""
    print(f"\n{violations['model']} on {violations['dataset']}:")
    print(f"  Total samples:           {violations['total_samples']}")
    print(f"  Samples with violations: {violations['total_violations']}")
    print(f"  Violation rate:          {violations['violation_rate']:.2f}%")
    
    if comparison:
        print(f"\n  Improvement over Baseline:")
        print(f"    Absolute: {comparison['absolute_improvement']:.2f} percentage points")
        print(f"    Relative: {comparison['relative_improvement_pct']:.2f}%")


def create_cross_dataset_summary(all_fuzzy_metrics=None, all_violations=None):
    """Create comprehensive cross-dataset comparison tables."""
    results = {}
    
    if all_fuzzy_metrics:
        fuzzy_data = []
        for key, metrics in all_fuzzy_metrics.items():
            fuzzy_data.append({
                'Dataset': metrics['dataset'],
                'Model': metrics['model'],
                'Fuzzy Loss': metrics['fuzzy_loss'],
                'Total Loss': metrics['total_loss']
            })
        results['fuzzy_comparison'] = pd.DataFrame(fuzzy_data)
    
    if all_violations:
        violation_data = []
        for key, viols in all_violations.items():
            violation_data.append({
                'Dataset': viols['dataset'],
                'Model': viols['model'],
                'Violation Rate (%)': viols['violation_rate'],
                'Violations': viols['total_violations'],
                'Total Samples': viols['total_samples']
            })
        results['violation_comparison'] = pd.DataFrame(violation_data)
    
    return results

def filter_dataset_by_labels(dataset, valid_labels):
    """Filter dataset to only include samples with labels in valid_labels"""
    valid_indices = []
    
    # Iterate through the dataset and collect indices with valid labels
    for idx in range(len(dataset)):
        idx, _, label = dataset[idx]  # Unpack: (idx, image, (concepts, label))
        if label not in valid_labels:
            valid_indices.append(idx)
    
    return Subset(dataset, valid_indices)
