from omegaconf import OmegaConf
from typing import List, Union, Dict, Any

from config import CBMTrainerConfig, StandardTrainerConfig

def load_config(config_path, configClass = CBMTrainerConfig, overrides: Union[List[str], Dict[str, Any], None] = None):
    """
    Load the configuration from a YAML file.
    """
    # Load YAML
    cfg_yaml = OmegaConf.load(config_path)
    cfg_structured = OmegaConf.structured(configClass)
    cfg = OmegaConf.merge(cfg_structured, cfg_yaml)
    cfg = OmegaConf.to_object(cfg)

    if overrides is not None:
        overrides_cfg = None
        if isinstance(overrides, list):
            # Create a DictConfig from a dot-list (e.g., ["model.hidden_size=512", "lr=0.001"])
            overrides_cfg = OmegaConf.from_dotlist(overrides)
        elif isinstance(overrides, dict):
            # Create a DictConfig from a standard dictionary
            overrides_cfg = OmegaConf.create(overrides)
        
        if overrides_cfg is not None:
            cfg = OmegaConf.merge(cfg, overrides_cfg)
            cfg = OmegaConf.to_object(cfg)

    cfg.resolve()
    return cfg