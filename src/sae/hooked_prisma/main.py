import torch
import os 
import sys 
from typing import Any, cast
import argparse 
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae.hooked_prisma.vision_config import VisionModelRunner

from sae.hooked_prisma.vision_activations_store import VisionActivationsStore
import json 
import csv
import torch
from vit_prisma.models.base_vit import HookedViT
from sae.language.sae_group import SAEGroup
from PIL import Image


from typing import Any, cast

import torch
import wandb
from torch.optim import Adam
from tqdm import tqdm

from sae.hooked_prisma.vision_evals import run_evals_vision
from sae.hooked_prisma.geometric_median import compute_geometric_median
from sae.language.optim import get_scheduler
from sae.language.sae_group import SAEGroup
from collections import defaultdict

def train_sae_on_vision_model(
    model: HookedViT,
    sae_group: SAEGroup,
    activation_store: VisionActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    #TODO hacky fix this
    total_training_tokens = cfg.total_training_images*cfg.context_size
    # batch size is number of tokens in a batcH??
    total_training_steps = total_training_tokens//batch_size
    n_training_steps = 0
    n_training_images = 0
    log_feature_sparsity = None

    checkpoint_thresholds = []
    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // n_checkpoints)
        )[1:]

    # things to store for each sae:
    # act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizer, scheduler,
    num_saes = len(sae_group)
    # track active features


    #TODO kinda seems like it could be handled in the group or something but w/e
    act_freq_scores = [
        torch.zeros(
            cast(int, sparse_autoencoder.cfg.d_sae),
            device=sparse_autoencoder.cfg.device,
        )
        for sparse_autoencoder in sae_group
    ]
    n_forward_passes_since_fired = [
        torch.zeros(
            cast(int, sparse_autoencoder.cfg.d_sae),
            device=sparse_autoencoder.cfg.device,
        )
        for sparse_autoencoder in sae_group
    ]
    n_frac_active_tokens = [0 for _ in range(num_saes)]

    optimizer = [Adam(sae.parameters(), lr=sae.cfg.lr) for sae in sae_group]
    scheduler = [
        get_scheduler(
            sae.cfg.lr_scheduler_name,
            optimizer=opt,
            warm_up_steps=sae.cfg.lr_warm_up_steps,
            training_steps=total_training_steps,
            lr_end=sae.cfg.lr / 10,  # heuristic for now.
        )
        for sae, opt in zip(sae_group, optimizer)
    ]

    all_layers = sae_group.cfg.hook_point_layer
    if not isinstance(all_layers, list):
        all_layers = [all_layers]

    # compute the geometric median of the activations of each layer

    geometric_medians = {}
    # extract all activations at a certain layer and use for sae initialization
    for sae in sae_group:
        hyperparams = sae.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
        if hyperparams.b_dec_init_method == "geometric_median":
            layer_acts = activation_store.storage_buffer.detach()[:, sae_layer_id, :]

            # get geometric median of the activations if we're using those.
            if sae_layer_id not in geometric_medians:

                median = compute_geometric_median(
                    layer_acts, maxiter=200,
                ).median
                geometric_medians[sae_layer_id] = median
            sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
        elif hyperparams.b_dec_init_method == "mean":
            layer_acts = activation_store.storage_buffer.detach().cpu()[:, sae_layer_id, :]

            sae.initialize_b_dec_with_mean(layer_acts)
        sae.train()

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_images < total_training_tokens:
      
        # Do a training step.
        layer_acts = activation_store.next_batch()
        n_training_images += batch_size

        # init these here to avoid uninitialized vars
        mse_loss = torch.tensor(0.0)
        l1_loss = torch.tensor(0.0)

        for (
            i,
            (sparse_autoencoder),
        ) in enumerate(sae_group):
            assert sparse_autoencoder.cfg.d_sae is not None  # keep pyright happy
            hyperparams = sparse_autoencoder.cfg
            layer_id = all_layers.index(hyperparams.hook_point_layer)
            sae_in = layer_acts[:, layer_id, :]

            sparse_autoencoder.train()
            # Make sure the W_dec is still zero-norm
            sparse_autoencoder.set_decoder_norm_to_unit_norm()

            # log and then reset the feature sparsity every feature_sampling_window steps
            if (n_training_steps + 1) % feature_sampling_window == 0:
                feature_sparsity = act_freq_scores[i] / n_frac_active_tokens[i]
                log_feature_sparsity = (
                    torch.log10(feature_sparsity + 1e-10).detach().cpu()
                )

                if use_wandb:
                    suffix = wandb_log_suffix(sae_group.cfg, hyperparams)
                    wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                    wandb.log(
                        {
                            f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
                            f"plots/feature_density_line_chart{suffix}": wandb_histogram,
                            f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5)
                            .sum()
                            .item(),
                            f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6)
                            .sum()
                            .item(),
                        },
                        step=n_training_steps,
                    )

                act_freq_scores[i] = torch.zeros(
                    sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
                )
                n_frac_active_tokens[i] = 0

            scheduler[i].step()
            optimizer[i].zero_grad()

            ghost_grad_neuron_mask = (
                n_forward_passes_since_fired[i]
                > sparse_autoencoder.cfg.dead_feature_window
            ).bool()

            # Forward and Backward Passes
            (
                sae_out,
                feature_acts,
                loss,
                mse_loss,
                l1_loss,
                ghost_grad_loss,
            ) = sparse_autoencoder(
                sae_in,
                ghost_grad_neuron_mask,
            )
            did_fire = (feature_acts > 0).float().sum(-2) > 0
            n_forward_passes_since_fired[i] += 1
            n_forward_passes_since_fired[i][did_fire] = 0

            with torch.no_grad():
                # Calculate the sparsities, and add it to a list, calculate sparsity metrics
                act_freq_scores[i] += (feature_acts.abs() > 0).float().sum(0)
                n_frac_active_tokens[i] += batch_size
                feature_sparsity = act_freq_scores[i] / n_frac_active_tokens[i]

                if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                    # metrics for currents acts
                    l0 = (feature_acts > 0).float().sum(-1).mean()
                    current_learning_rate = optimizer[i].param_groups[0]["lr"]

                    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                    total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
                    explained_variance = 1 - per_token_l2_loss / total_variance

                    suffix = wandb_log_suffix(sae_group.cfg, hyperparams)
                    wandb.log(
                        {
                            # losses
                            f"losses/mse_loss{suffix}": mse_loss.item(),
                            f"losses/l1_loss{suffix}": l1_loss.item()
                            / sparse_autoencoder.l1_coefficient,  # normalize by l1 coefficient
                            f"losses/ghost_grad_loss{suffix}": ghost_grad_loss.item(),
                            f"losses/overall_loss{suffix}": loss.item(),
                            # variance explained
                            f"metrics/explained_variance{suffix}": explained_variance.mean().item(),
                            f"metrics/explained_variance_std{suffix}": explained_variance.std().item(),
                            f"metrics/l0{suffix}": l0.item(),
                            # sparsity
                            f"sparsity/mean_passes_since_fired{suffix}": n_forward_passes_since_fired[
                                i
                            ]
                            .mean()
                            .item(),
                            f"sparsity/dead_features{suffix}": ghost_grad_neuron_mask.sum().item(),
                            f"details/current_learning_rate{suffix}": current_learning_rate,
                            "details/n_training_images": n_training_images,
                        },
                        step=n_training_steps,
                    )

                # record loss frequently, but not all the time.
                if use_wandb and (
                    (n_training_steps + 1) % (wandb_log_frequency * 10) == 0
                ):
                    sparse_autoencoder.eval()
                    suffix = wandb_log_suffix(sae_group.cfg, hyperparams)
                    run_evals_vision(
                        sparse_autoencoder,
                        activation_store,
                        model,
                        n_training_steps,
                        suffix=suffix,
                    )
                    sparse_autoencoder.train()

            loss.backward()
            sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
            optimizer[i].step()

        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_images > checkpoint_thresholds[0]:
            path = f"{sae_group.cfg.checkpoint_path}/{n_training_images}_{sae_group.get_name()}.pt"
            for sae in sae_group:
                sae.set_decoder_norm_to_unit_norm()
            sae_group.save_model(path)

            log_feature_sparsity_path = f"{sae_group.cfg.checkpoint_path}/{n_training_images}_{sae_group.get_name()}_log_feature_sparsity.pt"
            log_feature_sparsity = []
            for sae_id in range(len(sae_group)):
                feature_sparsity = (
                    act_freq_scores[sae_id] / n_frac_active_tokens[sae_id]
                )
                log_feature_sparsity.append(
                    torch.log10(feature_sparsity + 1e-10).detach().cpu()
                )
            torch.save(log_feature_sparsity, log_feature_sparsity_path)

            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if sae_group.cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sae_group.get_name()}",
                    type="model",
                    metadata=dict(sae_group.cfg.__dict__),
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)

                sparsity_artifact = wandb.Artifact(
                    f"{sae_group.get_name()}_log_feature_sparsity",
                    type="log_feature_sparsity",
                    metadata=dict(sae_group.cfg.__dict__),
                )
                sparsity_artifact.add_file(log_feature_sparsity_path)
                wandb.log_artifact(sparsity_artifact)

                ###############

        n_training_steps += 1
        pbar.set_description(
            f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
        )
        pbar.update(batch_size)

    # save sae group to checkpoints folder
    path = f"{sae_group.cfg.checkpoint_path}/final_{sae_group.get_name()}.pt"
    for sae in sae_group:
        sae.set_decoder_norm_to_unit_norm()
    sae_group.save_model(path)

    if sae_group.cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{sae_group.get_name()}",
            type="model",
            metadata=dict(sae_group.cfg.__dict__),
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])

    # need to fix this
    log_feature_sparsity_path = f"{sae_group.cfg.checkpoint_path}/final_{sae_group.get_name()}_log_feature_sparsity.pt"
    log_feature_sparsity = []
    for sae_id in range(len(sae_group)):
        feature_sparsity = act_freq_scores[sae_id] / n_frac_active_tokens[sae_id]
        log_feature_sparsity.append(
            torch.log10(feature_sparsity + 1e-10).detach().cpu()
        )
    torch.save(log_feature_sparsity, log_feature_sparsity_path)

    if sae_group.cfg.log_to_wandb:
        sparsity_artifact = wandb.Artifact(
            f"{sae_group.get_name()}_log_feature_sparsity",
            type="log_feature_sparsity",
            metadata=dict(sae_group.cfg.__dict__),
        )
        sparsity_artifact.add_file(log_feature_sparsity_path)
        wandb.log_artifact(sparsity_artifact)

    return sae_group


def wandb_log_suffix(cfg: Any, hyperparams: Any):
    # Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix



class ImageNetValidationDataset(torch.utils.data.Dataset):
        def __init__(self, images_dir, imagenet_class_index, validation_labels,  transform=None):
            self.images_dir = images_dir
            self.transform = transform
            self.labels = {}


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

            return image, label_i



    
def setup(checkpoint_path,imagenet_path, num_workers=0, pretrained_path=None, expansion_factor = 64, layers=9, context_size=50, model_name='vit_base_patch32_224'):

    # assuming the same structure as here: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    imagenet_train_path = os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/train")
    imagenet_val_path  =os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/val")
    imagenet_val_labels = os.path.join(imagenet_path, "LOC_val_solution.csv")
    imagenet_label_strings = os.path.join(imagenet_path, "LOC_synset_mapping.txt" )

    cfg = VisionModelRunner(
        #TODO expose more 
    # Data Generating Function (Model + Training Distibuion)
   # model_name = "gpt2-small",
    model_name = model_name, #
    hook_point = "blocks.{layer}.hook_resid_pre",
    hook_point_layer = layers, # 
    d_in = 768,
    #dataset_path = "Skylion007/openwebtext", #
    #is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = expansion_factor, # online weights used 32! 
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = context_size, # TODO should be auto 
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    #total_training_tokens = 1_000_000 * 300, #
    total_training_images = 1_300_000*5,


    store_batch_size = 32, # num images
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
   # dead_feature_threshold = 1e-6, # did not apper to be used in either future, (is it a different method than window?)
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "vit_sae_training", #
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = checkpoint_path, # #TODO 
    dtype = torch.float32,

    #loading
    from_pretrained_path = pretrained_path
    )



    if cfg.from_pretrained_path is not None:
        sae_group = SAEGroup.load_from_pretrained(cfg.from_pretrained_path)
    else:
        sae_group = SAEGroup(cfg)


    model = HookedViT.from_pretrained(cfg.model_name)
    model.to(cfg.device)


    import torchvision
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    


    imagenet1k_data = torchvision.datasets.ImageFolder(imagenet_train_path, transform=data_transforms)




    
    imagenet1k_data_val = ImageNetValidationDataset(imagenet_val_path,imagenet_label_strings, imagenet_val_labels ,data_transforms)




    activations_loader = VisionActivationsStore(
        cfg,
        model,
        imagenet1k_data,

        num_workers=num_workers,
        eval_dataset=imagenet1k_data_val,
    )


  


    return cfg , model, activations_loader, sae_group


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", 
                        required=True,
                        help="folder where you will save checkpoints"
                        )
    parser.add_argument('--imagenet_path', required=True, help='folder containing imagenet1k data organized as follows: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description')
    parser.add_argument("--eval",
                        action="store_true",
                        help="run evals")
    
    parser.add_argument("--expansion_factor",
                        type=int,
                        default=64)
    parser.add_argument("--context_size",
                        type=int,
                        default=50)
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=4)
    parser.add_argument("--model_name",
                        default="vit_base_patch32_224")
    parser.add_argument("--layers",
                    type=int,
                    nargs="+",
                    default=[9])


    parser.add_argument('--load', type=str, default=None, help='Pretrained SAE path')
    args = parser.parse_args()

    #TODO make note of when/if this changes... best to just make a new version tbh
    cfg ,model, activations_loader, sae_group = setup(args.checkpoint_path,args.imagenet_path, pretrained_path=args.load, expansion_factor=args.expansion_factor, layers=args.layers, context_size=args.context_size, model_name=args.model_name ,num_workers=args.num_workers)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)

    if args.eval:   
        for (
            i,
            (sae),
        ) in enumerate(sae_group):
            sae.eval()
            hyperparams = sae.cfg
            suffix = wandb_log_suffix(sae_group.cfg, hyperparams)
            run_evals_vision(
                sae,
                activations_loader,
                model,
                0,
                suffix=suffix,
            )
    else:
        # train SAE
        train_sae_on_vision_model(
            model,
            sae_group,
            activations_loader,
            n_checkpoints=cfg.n_checkpoints,
            batch_size=cfg.train_batch_size,
            feature_sampling_window=cfg.feature_sampling_window,
            use_wandb=cfg.log_to_wandb,
            wandb_log_frequency=cfg.wandb_log_frequency,
        )

    if cfg.log_to_wandb:
        wandb.finish()