from numpy.random import seed as set_numpy_seed
from torch import manual_seed as set_torch_seed
from torch.cuda import manual_seed_all as set_torch_cuda_seed
from random import seed as set_random_seed
from torch.backends import cudnn

from config import load_config

from config import CBMTrainerConfig
from models.architectures import CBMSequentialEfficientNetFCN, EfficientNetv2
from models.trainer.cbm_trainer import CBMTrainer
from models.trainer.standard_trainer import StandardTrainer
from config.standard_trainer_config import StandardTrainerConfig

def set_reproducibility_seed(seed):
    print(seed)
    set_random_seed(seed)
    set_numpy_seed(seed)
    set_torch_seed(seed)
    set_torch_cuda_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def initialize_cbm_experiment(seed, device_no, config_file):
    print("Initializing CBM experiment...")
    config = load_config(
        config_file, configClass=CBMTrainerConfig, overrides=[f"seed={seed}", f"device_no={device_no}"]
    )
    print("seed inside config", config.seed)
    train_loader, val_loader, test_loader = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).get_dataloaders()
    model = CBMSequentialEfficientNetFCN(config)
    print("got until before trainer")
    trainer = CBMTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    return train_loader, val_loader, test_loader, model, trainer, config

def initialize_standard_cnn_experiment(seed, device_no, config_file):
    print("Initializing standard CNN experiment...")
    config = load_config(
        config_file, configClass=StandardTrainerConfig, overrides=[f"seed={seed}", f"device_no={device_no}"]
    )
    train_loader, val_loader, test_loader = config.dataset.factory(
        seed=config.seed, config=config.dataset
    ).get_dataloaders()
    model = EfficientNetv2(config.dataset.n_labels).to(config.device)
    trainer = StandardTrainer(
        config=config.training,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.device,
    )
    return train_loader, val_loader, test_loader, model, trainer, config

def initialize_experiment(seed, device_no, config_file, model_type):
    set_reproducibility_seed(seed)
    if model_type == "reqaware" or model_type == "vanilla_cbm":
        return initialize_cbm_experiment(seed, device_no, config_file)
    elif model_type == "baseline_cnn":
        return initialize_standard_cnn_experiment(seed, device_no, config_file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")   
    

def load_cbm_model(config, directory, concept_train_loader, concept_val_loader, concept_test_loader,):
    config.concept_predictor.pretrained_weights = directory
    trainer = CBMTrainer(
        config=config,
        model=CBMSequentialEfficientNetFCN(config),
        train_loader=concept_train_loader,
        val_loader=concept_val_loader,
        test_loader=concept_test_loader,
    )
    return trainer

def load_standard_model(config, directory, train_loader, val_loader, test_loader):
    config.training.pretrained_weights = directory
    trainer = StandardTrainer(
        config=config.training,
        model=EfficientNetv2(config.dataset.n_labels).to(config.device),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.device,
    )
    return trainer