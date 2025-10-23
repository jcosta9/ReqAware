# Leveraging Domain Requirements in Concept Based Models via Differentiable Fuzzy Logic

## 🧠 Overview

This repository provides an implementation of the paper *"Leveraging Domain Requirements in Concept Based Models via Differentiable Fuzzy Logic"*.  The goal of this project is to provide full access to the research results, by following the principles of transparency, reproducibility and replicability.

> **Abstract:**  
> When designing machine learning (ML) enabled software systems, it is often assumed that the training dataset implicitly captures the full domain requirements stated by the system's specification. Guaranteeing this assumption is challenging, as models frequently learn spurious correlations rather than the true underlying domain logic, leading to unreliable system behavior. Prior work addresses this by integrating domain requirements directly into the model's loss function via Fuzzy First-Order Logic (FOL) constraints, \ie differentiable mathematical expressions derived from human-readable domain rules. However, the expressiveness and hence effectiveness of these approaches are often limited, as they must directly map complex logic onto abstract features or final labels, lacking a semantically meaningful intermediate representation for precise control. We resolve this limitation using the Fuzzy Concept Bottleneck Model (ReqAware). By leveraging the CBM's intermediate layer of human-interpretable concepts, we can specify and translate complex domain requirements via differentiable fuzzy logic constraints. This concept-based approach offers flexibility and precision in enforcing knowledge, transforming the interpretability layer into a verifiable, logically consistent interface. We evaluate our approach in the traffic sign domain using the \gls{gtsrb} dataset by eliciting a set of domain requirements and training the \gls{cbm} using them. We demonstrate how our approach can improve the \gls{cbm}'s predictive performance and its satisfaction of the requirements.

---

## 🏗️ Repository Structure

```
<project-root>/
├── files/                         # FOlder for experiment configurations and models
│   ├── configs                    # stores yaml files with hyperparameters for experiments
│   └── models                     # stores trained models saved for reproducibility purposes
├── Makefile                       # Makefile for convenience
├── notebooks/                     # Graphical results and analysis
│   ├── RQ1.ipynb                  # Notebook for reproducing RQ1 results
│   └── RQ2_3.ipynb                # Notebook for reproducing RQ2 and RQ3 results
├── README.md
├── requirements.txt
└── src/                           # Code base used to obtain the results
    ├── experiments/               # stores experiments logs and results
    │   └── RQ                     # results for the research questions
    │       ├── Results/
    │       │   ├── plots/         # plots used in the paper
    ├── config/                    # Code responsible for reading and parsing yaml files into usable config objects
    ├── data_access/               # Code for handling datasets
    │   ├── concepts/              # Store classes used for reading concept aware datasets
    │   ├── datasets/              # Factory pattern for reading datasets
    │   ├── preprocess/            # Functions for augmenting and enhancing images
    │   └── registry.py            # Maps dataset names with their implementation
    ├── models                     # Code for implementing the models
    │   ├── architectures/         # Backbone architectures
    │   ├── loss/                  # Implementation of fuzzy rules as loss funtions
    │   ├── registries/            # diverse mappings
    │   ├── trainer/               # Code for handling the training of models
    │   └── utils/                 # Utility functions
    ├── rule_eval/                 # Code for checking rule violations
    ├── hyperparameter_opt.py      # Functions used in the hyperparameter tuning proccess
    ├── reproduce_experiments.py   # Code used to generate the models used in the evaluation
    ├── collect_results.py         # after generating the models, this code collects the results.
    ├── train_cbm.py               # Script for training a CBM
    ├── train.py                   # Script for training a baseline CNN
    └── utils/                     # Utility functions
````

---

## ⚙️ Installation

1. Clone this repository

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate       # (Linux/Mac)
   venv\Scripts\activate          # (Windows)
   pip install -r requirements.txt
   ```

3. Download or prepare datasets
  - [German Traffic Sign Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html)
  - [Belgium Traffic Sign Dataset (BTSD)](https://btsd.ethz.ch/shareddata/)
---

## 🚀 Usage

### Train a Baseline CNN model

```bash
python src/train.py --config_file /path/to/config.yaml
```

### Train a ReqAware model

```bash
python src/train_cbm.py --config_file /path/to/config.yaml
```

### Evaluate the model

```bash
make reproduce-experiments
```
Afterwards, run files Notebooks/RQ1.ipynb and Notebooks/RQ2_3.ipynb

---


## 🔍 Extended Work (Optional)

This codebase can be easily tweaked to work with new datasets and new sets of requirements. For example:

- New datasets can be created by developing a data factory to handle its reading and adding them to the registry.
- New requirements can be implemented by using the Fuzzy transformations inside src/models/loss/fuzzy_transformations.py and by placing them in custom_rules.py 

---

## 📦 Dependencies

* Python >= 3.8
* torch == 2.9.0
* NumPy == 1.23.5 

See [`requirements.txt`](./requirements.txt) for the full list.

---
