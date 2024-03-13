from dataclasses import dataclass
from sae.language.config import LanguageModelSAERunnerConfig


@dataclass
class VisionModelRunner(LanguageModelSAERunnerConfig):
    total_training_images:int = 1_000_000
