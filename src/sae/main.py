














import torchvision
from transformers import CLIPProcessor

import os
import signal
from typing import Any, Optional, cast
from vit_prisma.models.base_vit import HookedViT

import numpy as np
import torch
import yaml
import argparse
import wandb
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from sae_lens import __version__

# from sae_lens.training.sae_group import SparseAutoencoderDictionary
# from sae_lens.training.sparse_autoencoder import (
#     SAE_CFG_PATH,
#     SAE_WEIGHTS_PATH,
#     SPARSITY_PATH,
# )
# from sae_lens.training.train_sae_on_language_model import (
#     SAETrainContext, SAETrainingRunState, TrainSAEGroupOutput,
#      get_total_training_tokens, _wandb_log_suffix, _build_train_context,
#      _init_sae_group_b_decs, _update_sae_lens_training_version, _train_step, _build_train_step_log_dict,
#      TRAINING_RUN_STATE_PATH, SAE_CONTEXT_PATH
# )

import re

from sae.vision_evals import run_evals_vision
from sae.vision_config import VisionModelRunnerConfig
from sae.vision_activations_store import VisionActivationsStore
import csv



import json
import logging
import os
import signal
from typing import Any, cast

import torch
import wandb
from safetensors.torch import save_file
from transformer_lens.hook_points import HookedRootModule

from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.load_model import load_model
from sae_lens.sae import SAE_CFG_PATH, SAE_WEIGHTS_PATH, SPARSITY_PATH
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig, handle_config_defaulting, read_sae_from_disk, DTYPE_MAP, SAE_CFG_PATH, SAE_WEIGHTS_PATH
import json 

class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):
    raise InterruptedException()



# same as saelens but replaceing run_evals with run_evals_vision. Also fixes issue with load from pretrain
class VisionSAETrainer(SAETrainer):
    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        if (self.n_training_steps + 1) % (
            self.cfg.wandb_log_frequency * self.cfg.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            
            #TODO update for newer features. Fix for clip
            run_evals_vision(
                    self.sae,
                    self.activation_store,
                    self.model,
                    self.n_training_steps,
                )

            extra_eval_metrics = {}
            W_dec_norm_dist = self.sae.W_dec.norm(dim=1).detach().cpu().numpy()
            extra_eval_metrics["weights/W_dec_norms"] = wandb.Histogram(W_dec_norm_dist)  # type: ignore

            if self.sae.cfg.architecture == "standard":
                b_e_dist = self.sae.b_enc.detach().cpu().numpy()
                extra_eval_metrics["weights/b_e"] = wandb.Histogram(b_e_dist)  # type: ignore
            elif self.sae.cfg.architecture == "gated":
                b_gate_dist = self.sae.b_gate.detach().cpu().numpy()
                extra_eval_metrics["weights/b_gate"] = wandb.Histogram(b_gate_dist)  # type: ignore
                b_mag_dist = self.sae.b_mag.detach().cpu().numpy()
                extra_eval_metrics["weights/b_mag"] = wandb.Histogram(b_mag_dist)  # type: ignore

            wandb.log(
                extra_eval_metrics,
                step=self.n_training_steps,
            )
            self.sae.train()


# normally handled in TrainingSAE but I was running into an issue which I fix below (alternative would be to modify TraniningSAEConfig)
def load_sae(
        path: str,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> TrainingSAE:

        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
            dtype=DTYPE_MAP[dtype],
        )


        # added this code which removes any unwanted keys 
        dummy_cfg = VisionModelRunnerConfig().get_vision_training_sae_cfg_dict()
        for key in list(cfg_dict.keys()):
            if key not in dummy_cfg.keys():
                del cfg_dict[key]
        
        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        sae = TrainingSAE(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae


# same as saelens but custom init and using VisionSAETrainer
class VisionSAETrainingRunner(SAETrainingRunner):
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: VisionModelRunnerConfig
    model: HookedRootModule
    sae: TrainingSAE
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: VisionModelRunnerConfig,
        model: HookedViT,
        activations_store:VisionActivationsStore,
        sae:TrainingSAE,

    ):
        assert cfg.eval_batch_size_prompts is None, "eval_batch_size_prompts not implemented for vision! (argument is being used for hacky override)"
        cfg.eval_batch_size_prompts = "override"
        self.cfg = cfg

        self.model = model
        self.activations_store = activations_store

        self.sae = sae 
        if cfg.from_pretrained_path is  None:
            self._init_sae_group_b_decs()



    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = VisionSAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae



def get_imagenet_index_to_name(imagenet_path):
    ind_to_name = {}

    with open( os.path.join(imagenet_path, "LOC_synset_mapping.txt" ), 'r') as file:
        # Iterate over each line in the file
        for line_num, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            label = parts[1].split(',')[0]
            ind_to_name[line_num] = label
    return ind_to_name


class ImageNetValidationDataset(torch.utils.data.Dataset):
        def __init__(self, images_dir, imagenet_class_index, validation_labels,  transform=None, return_index=False):
            self.images_dir = images_dir
            self.transform = transform
            self.labels = {}
            self.return_index = return_index


            # load label code to index
            self.label_to_index = {}
    
            with open(imagenet_class_index, 'r') as file:
                # Iterate over each line in the file
                for line_num, line in enumerate(file):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    code = parts[0]
                    self.label_to_index[code] = line_num


            # load image name to label code
            self.image_name_to_label = {}

            # Open the CSV file for reading
            with open(validation_labels, mode='r') as csv_file:
                # Create a CSV reader object
                csv_reader = csv.DictReader(csv_file)
                
                # Iterate over each row in the CSV
                for row in csv_reader:
                    # Split the PredictionString by spaces and take the first element
                    first_prediction = row['PredictionString'].split()[0]
                    # Map the ImageId to the first part of the PredictionString
                    self.image_name_to_label[row['ImageId']] = first_prediction

            self.image_names = list(os.listdir(self.images_dir))

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, idx):

            img_path = os.path.join(self.images_dir, self.image_names[idx])
           # print(img_path)
            image = Image.open(img_path).convert('RGB')

            img_name = os.path.basename(os.path.splitext(self.image_names[idx])[0])

            label_i = self.label_to_index[self.image_name_to_label[img_name]]

            if self.transform:
                image = self.transform(image)

            if self.return_index:
                return image, label_i, idx
            else:
                return image, label_i


def setup(cfg, setup_args, legacy_load=False):
    # assuming the same structure as here: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    imagenet_train_path = os.path.join(setup_args['imagenet_path']['value'], "ILSVRC/Data/CLS-LOC/train")
    imagenet_val_path  =os.path.join(setup_args['imagenet_path']['value'], "ILSVRC/Data/CLS-LOC/val")
    imagenet_val_labels = os.path.join(setup_args['imagenet_path']['value'], "LOC_val_solution.csv")
    imagenet_label_strings = os.path.join(setup_args['imagenet_path']['value'], "LOC_synset_mapping.txt" )
   
    clip_processor = CLIPProcessor.from_pretrained(cfg.model_name)
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        #TODO for clip only 
        torchvision.transforms.Normalize(mean=clip_processor.image_processor.image_mean,
                         std=clip_processor.image_processor.image_std),
    ])

    imagenet1k_data = torchvision.datasets.ImageFolder(imagenet_train_path, transform=data_transforms)
    imagenet1k_data_val = ImageNetValidationDataset(imagenet_val_path,imagenet_label_strings, imagenet_val_labels ,data_transforms)

    
    cfg.training_tokens = int(len(imagenet1k_data)*setup_args['num_epochs']) * cfg.context_size

     #TODO support cfg.resume
    if cfg.from_pretrained_path is not None:

        if legacy_load:
            from sae.legacy_load import load_legacy_pt_file
            sae = load_legacy_pt_file(cfg.from_pretrained_path)
        else:
            sae = load_sae(
                cfg.from_pretrained_path, cfg.device
            )
    else:
        sae = TrainingSAE(
            TrainingSAEConfig.from_dict(
                cfg.get_vision_training_sae_cfg_dict(),
            )
        )

    model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True)
    model.to(cfg.device)



    

    activations_store = VisionActivationsStore(
        cfg,
        model,
        imagenet1k_data,

        num_workers=setup_args['num_workers'],
        eval_dataset=imagenet1k_data_val,
    )

    return model, activations_store, sae


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="path to training config")
    parser.add_argument("--legacy_load", action="store_true", help="use to load models trained with a older version of this code")
    args = parser.parse_args()

    setup_args = yaml.safe_load(open(args.config, 'r'))['setup_arguments']

    cfg = VisionModelRunnerConfig.from_yaml(args.config)

    model, activations_store, sae = setup(cfg, setup_args, legacy_load=args.legacy_load)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)


    # train SAE
        

    VisionSAETrainingRunner(cfg,model,activations_store,sae).run( )
    


