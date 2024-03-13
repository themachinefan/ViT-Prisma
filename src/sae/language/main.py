import torch
import os 
import sys 
from typing import Any, cast

import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae.language.config import LanguageModelSAERunnerConfig

# from sae.language.activation_store import ActivationStore
from sae.language.train_sae_on_language_model import train_sae_on_language_model
from sae.language.utils import LMSparseAutoencoderSessionloader


cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    hook_point = "blocks.2.hook_resid_pre",
    hook_point_layer = 2,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = 64,
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    total_training_tokens = 1_000_000 * 300,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats_sae_training_gpt2",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = "F:\ViT-Prisma_fork\data\sae_checkpoints",
    dtype = torch.float32,
    )



if cfg.from_pretrained_path is not None:
    (
        model,
        sparse_autoencoder,
        activations_loader,
    ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
        cfg.from_pretrained_path
    )
    cfg = sparse_autoencoder.cfg
else:
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

if cfg.log_to_wandb:
    wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)


print("HI", len(sparse_autoencoder.autoencoders))
# train SAE
sparse_autoencoder = train_sae_on_language_model(
    model,
    sparse_autoencoder,
    activations_loader,
    n_checkpoints=cfg.n_checkpoints,
    batch_size=cfg.train_batch_size,
    feature_sampling_window=cfg.feature_sampling_window,
    dead_feature_threshold=cfg.dead_feature_threshold,
    use_wandb=cfg.log_to_wandb,
    wandb_log_frequency=cfg.wandb_log_frequency,
)

if cfg.log_to_wandb:
    wandb.finish()