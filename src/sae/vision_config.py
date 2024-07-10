from dataclasses import dataclass
from sae_lens.config import LanguageModelSAERunnerConfig
import yaml
from typing import Any 
import re 

@dataclass
class VisionModelRunnerConfig(LanguageModelSAERunnerConfig):
    store_batch_size:int =32 # num images

    @classmethod
    def from_yaml(cls, path:str):
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return cls(**config_dict['arguments'])
    

    def get_vision_training_sae_cfg_dict(self) -> dict[str, Any]:
        new_config = {
            **self.get_training_sae_cfg_dict(),
        }
        new_config["model_name"] = re.sub(r'[^a-zA-Z0-9._-]', '_', new_config["model_name"]) # otherwise wandb will crash when creating artifacts since model name is used. Alternatively could override SAE.get_name() but I think it's better to avoid another new class. 
        
        
        return new_config