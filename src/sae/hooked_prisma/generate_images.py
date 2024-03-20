from sae.hooked_prisma.main import setup
import os
import sys
import torch
import torch.nn.functional as F
import json
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import Dict
from pathlib import Path
import einops 
from functools import partial
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from einops import rearrange, repeat
import base64
from sae.hooked_prisma.main import ImageNetValidationDataset, get_imagenet_index_to_name
import torchvision
#temp
from sae.hooked_prisma.vision_config import VisionModelRunner
import cv2
torch.set_grad_enabled(False)
from dataclasses import dataclass

@dataclass
class FeatureInfo:
    id: int
    layer:int
    hit_count:float 


@torch.inference_mode()
# count how often a feature is activated
def count_activated_batch(
    images,
    model,
    autoencoders,
    threshold = -3,
    layers = None,
    device="cuda",
):
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    if layers is None:
        layers = list(range(len(autoencoders)))
    _, cache = model.run_with_cache(images)

    hit_counts = []
    for layer in layers:
        autoencoder = autoencoders[layer]
        _, feature_acts, _, _, _, _ = autoencoder(
            cache[autoencoder.cfg.hook_point]
        )
        feature_acts = einops.rearrange(feature_acts, "batch seq d_mlp -> (batch seq) d_mlp")
        feature_probability = feature_acts
        log_freq = (feature_probability + 1e-10).log10()



        activated_neurons = log_freq > threshold 
        hit_count = torch.count_nonzero(activated_neurons, dim=0)
        print(hit_count.shape)
        hit_counts.append(hit_count)

    hit_count = torch.stack(hit_counts,dim=0)
    
    return hit_count
def count_activated(
    dataloader,
    model,
    autoencoders,
    threshold = -3,
    layers = None,
    device = "cuda",
):
    total = None 
    for total_images, _, _ in tqdm(dataloader):
            total_images = total_images.to(device)
            new = count_activated_batch(total_images, model, autoencoders, threshold=threshold, layers=layers, device=device)

            if total is None:
                total = new
            else:
                total = total + new 
    return total 


def find_highest_activating_batch(
    images,
    model,
    autoencoders,
    feature_ids,
    k: int = 10,
): 
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    num_autoencoders,k, num_features 4,  (where 4 = batchid, patchid, val, predclass)  
    '''
    # Get the post activations from the clean run
    _, cache = model.run_with_cache(images)
    # sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid = sparse_autoencoder(
    #     cache[sparse_autoencoder.cfg.hook_point]
    # )
    
    largest_activations = []
    smallest_activations = []
    for cur_feature_ids, autoencoder in zip(feature_ids, autoencoders):
        post_reshaped = einops.rearrange( cache[autoencoder.cfg.hook_point], "batch seq d_mlp -> (batch seq) d_mlp")
        seq_len = cache[autoencoder.cfg.hook_point].shape[1]
        sae_in =  post_reshaped - autoencoder.b_dec 
        acts = einops.einsum(
                sae_in,
                autoencoder.W_enc[:, cur_feature_ids],
                "... d_in, d_in n -> ... n",
            )
        
        top_acts_values, top_acts_indices = acts.topk(k, dim=0)
        top_acts_batch = top_acts_indices // seq_len
        top_acts_seq = top_acts_indices % seq_len
        largest = torch.stack([top_acts_batch, top_acts_seq, top_acts_values], dim=-1)
        largest_activations.append(largest)

        top_acts_values, top_acts_indices = acts.topk(k, dim=0, largest=False)
        top_acts_batch = top_acts_indices // seq_len
        top_acts_seq = top_acts_indices % seq_len
        largest = torch.stack([top_acts_batch, top_acts_seq, top_acts_values], dim=-1)
        smallest_activations.append(largest)
    big = torch.stack(largest_activations, dim=0)
    small = torch.stack(smallest_activations, dim=0)
    return big ,  small
      #  to_return[feature_id]  = (torch.stack([top_acts_batch, top_acts_seq], dim=-1), top_acts_values)
    #return to_return 

def find_highest_activating(
    dataloader,
    model,
    autoencoders,
    feature_ids,
    device="cuda",
    k: int = 10,
): # num_autoencoders, num_features, k, 4
    biggest, smallest = None, None

    n, m, p = len(autoencoders), feature_ids.shape[1], 3
    n_indices = torch.arange(n).view(-1, 1, 1, 1).expand(n, k, m, p).to(device)
    m_indices = torch.arange(m).view(1, 1, -1, 1).expand(n, k, m, p).to(device)
    p_indices = torch.arange(p).view(1, 1, 1, -1).expand(n, k, m, p).to(device)
    for  total_images, _, total_indices in tqdm(dataloader):
        total_images = total_images.to(device)
        total_indices = total_indices.to(device)
        cur_biggest, cur_smallest = find_highest_activating_batch(total_images, model, autoencoders, feature_ids, k=k)
        # convert batch_id to dataset id
        cur_biggest[...,0] = total_indices[cur_biggest[...,0].long()]
        cur_smallest[...,0] = total_indices[cur_smallest[...,0].long()]
        # max/min the cur with total
        if biggest is None:
            biggest= cur_biggest
            smallest = cur_smallest
        else:
            combined_b = torch.cat([biggest, cur_biggest], dim=1)
            combined_s = torch.cat([smallest, cur_smallest], dim=1)

            _, inds = torch.topk(combined_b[...,2], k, dim=1)
          
            biggest = combined_b[n_indices, inds.unsqueeze(-1).expand(-1, -1, -1, p), m_indices, p_indices]
            _, inds = torch.topk(combined_s[...,2], k,dim=1,largest=False)

            smallest = combined_s[n_indices, inds.unsqueeze(-1).expand(-1, -1, -1, p), m_indices, p_indices]



    return einops.rearrange(biggest, "n k m p -> n m k p"),  einops.rearrange(smallest, "n k m p -> n m k p")


def generate_images(
        features_ids, 
        features_hit_count,
        highest_activations,
        imagenet_dataset,
        imagenet_index_to_name,
        autoencoder_layer_names,
        output_folder,
        header="",
        border_color=(0,0,0),
        patch_size=32,
):
    BID = 0
    PID = 1
    VAL = 2
    for i in range(features_ids.shape[0]):
        autoencoder_id = autoencoder_layer_names[i] 
        for ii, feature_id in enumerate(features_ids[i]):
            print(f"looking at {feature_id}")
            images = []
            gt_labels = []

            for iii in range(highest_activations.shape[2]):
                bid = int(highest_activations[i, ii, iii, BID].item())
                image, label, image_ind = imagenet_dataset[bid]

                assert image_ind == bid
                images.append(image)
                gt_labels.append(imagenet_index_to_name[label])
            

            #TODO title autoencoder id, feature id, hitcount, 
            grid_size = int(np.ceil(np.sqrt(len(images))))
            fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))

            fig.suptitle(f'{header}\nLayer {autoencoder_id}, Feature {feature_id}, hit count: {features_hit_count[i,ii]}')
            for ax in axs.flatten():
                ax.axis('off')
            for grid_i, (image_tensor, label, iii) in enumerate(zip(images, gt_labels, list(range(highest_activations.shape[2])))):
                pid, act_val = int(highest_activations[i, ii, iii, PID].item()), highest_activations[i, ii, iii, VAL].item()

                row = grid_i // grid_size
                col = grid_i % grid_size
                display = image_tensor.numpy().transpose(1, 2, 0)

                if pid == 0:
                    pass
                else:
                    pid_r = (pid-1)//(224//patch_size)
                    pid_c = (pid-1)%(224//patch_size)
                    
                    display[pid_r*patch_size:(pid_r+1)*patch_size, pid_c*patch_size:(pid_c+1)*patch_size,:] = display[pid_r*patch_size:(pid_r+1)*patch_size, pid_c*patch_size:(pid_c+1)*patch_size,:]*0.75 + 0.25*np.zeros_like(display[pid_r*patch_size:(pid_r+1)*patch_size, pid_c*patch_size:(pid_c+1)*patch_size,:])
                    display[pid_r*patch_size:(pid_r+1)*patch_size, pid_c*patch_size,0] = 1.0
                    display[pid_r*patch_size:(pid_r+1)*patch_size, (pid_c+1)*patch_size-1,0] = 1.0
                    display[pid_r*patch_size, pid_c*patch_size:(pid_c+1)*patch_size,0] =1.0
                    display[(pid_r+1)*patch_size-1, pid_c*patch_size:(pid_c+1)*patch_size,0]  =1.0
                #TODO save single values too?
           

                axs[row, col].imshow(display)
                axs[row, col].set_title(f"gt: {label} {act_val:0.03f} {'class token!' if pid==0 else ''}")  
                axs[row, col].axis('off')  

            #TODO SAVE
                            
            from io import BytesIO

            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Load the buffer as an image using OpenCV
            img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            # Specify the border size and color
            border_size = 5

            # Add the border to the image
            bordered_img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=border_color)

            cv2.imwrite(os.path.join(output_folder,f'layer_{autoencoder_id}_feature_{feature_id}.png'), bordered_img)

          #  fig.savefig(os.path.join(output_folder,f'layer_{autoencoder_id}_feature_{feature_id}.png'), dpi=300)

if __name__ == "__main__":
    use_cache = True
    batch_size = 64
    num_workers = 0
    device = "cuda" #not actually set up for cpu atm :P 
    checkpoint_path= "F:/ViT-Prisma_fork/data/vision_sae_checkpoints"
    imagenet_path = "F:/prisma_data/imagenet-object-localization-challenge"
    pretrained_path="F:/ViT-Prisma_fork/data/vision_sae_checkpoints/organized/130002944_sae_group_vit_base_patch32_224_blocks.-11.hook_resid_pre_24576.pt"
    output_path = "F:/ViT-Prisma_fork/data/feature_images"
    output_cache = os.path.join(output_path, "cache")
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(output_cache,exist_ok=True)




    

    #TODO these are saved in config probably, should load from there (in fact I think it already does). 
    # pretrained_layers = [9]
    # pretrained_expansion_factor = 32
    # pretrained_model_name = "vit_base_patch16_224"
    # pretrained_context_size = 197
    pretrained_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
    pretrained_expansion_factor = 32
    pretrained_model_name = "vit_base_patch32_224"
    pretrained_context_size = 50 

    cfg ,model, activations_loader, sae_group = setup(checkpoint_path=checkpoint_path, 
                                                    imagenet_path=imagenet_path ,
                                                        pretrained_path=pretrained_path, layers= pretrained_layers, expansion_factor=pretrained_expansion_factor,
                                                        model_name=pretrained_model_name, context_size=pretrained_context_size)

    for i, sae in enumerate(sae_group):
        hyp = sae.cfg
        print(
            f"{i}: Layer {hyp.hook_point_layer}, p_norm {hyp.lp_norm}, alpha {hyp.l1_coefficient}"
        )

    saes = sae_group.autoencoders
    for s in saes:
        print(cfg.hook_point)




    data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])


    # assuming the same structure as here: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    imagenet_val_path  =os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/val")
    imagenet_val_labels = os.path.join(imagenet_path, "LOC_val_solution.csv")
    imagenet_label_strings = os.path.join(imagenet_path, "LOC_synset_mapping.txt" )


    imagenet_data = ImageNetValidationDataset(imagenet_val_path,imagenet_label_strings, imagenet_val_labels ,data_transforms, return_index=True)


    dataloader = DataLoader(imagenet_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    imagenet_index_to_name = get_imagenet_index_to_name(imagenet_path)

    hit_count_path = os.path.join(output_cache, "hit_count.pt" )

    
    if os.path.exists(hit_count_path) and use_cache:
        hit_count = torch.load(hit_count_path)
        hit_count = hit_count.to(device)
    else:
        for thing, _, _ in tqdm(dataloader):
            print(thing)
        hit_count = count_activated(
            dataloader,
            model,
            saes,
            device=device)
        torch.save(hit_count, hit_count_path)
    
    print(hit_count.shape)
    top_amount = 20
    random_amount = 20
    top_vals, top_inds = hit_count.topk(top_amount, dim=1)
    


    random_indices = torch.stack([torch.randperm(hit_count.shape[1]).to(device)[:random_amount] for _ in range(hit_count.shape[0])])
    random_values = torch.gather(hit_count, 1, random_indices)

    print(top_inds.shape, random_indices.shape)
    all_indices = torch.cat([top_inds, random_indices], dim=1)


    num_images = 25
    print("TODO SAVE CACHE (could include loading hitcount)")
    highest_activating_path = os.path.join(output_cache, "highest_activating.pt" )

    if os.path.exists(highest_activating_path) and use_cache:
        highest_activation, lowest_activation = torch.load(highest_activating_path)
    else:
        highest_activation, lowest_activation = find_highest_activating(dataloader, model, saes, all_indices,k=num_images, device=device )
        torch.save([highest_activation,lowest_activation], highest_activating_path)

    ha_top, ha_rand = highest_activation[:, :top_amount], highest_activation[:, top_amount:]
    la_top, la_rand = lowest_activation[:, :top_amount], lowest_activation[:, top_amount:]


    for name, feature_ids, feature_hit_count, highest_activation, border_color in zip(["top_feature_high_activations","top_feature_low_activations","random_feature_high_activations","random_feature_low_activations"],
                                                                        [top_inds, top_inds, random_indices, random_indices],
                                                                        [top_vals,top_vals,random_values,random_values],
                                                                        [ha_top, la_top, ha_rand, la_rand],
                                                                   #     [(0.0,1.0,0.0), (0.,0.0,1.), (0.,1.,0.), (0.,0.,1.)]
                                                                        [(0,255,0), (0,0,255), (0,255,0), (0,0,255)]

                                                                        ):
        fold = os.path.join(output_path, name)
        os.makedirs(fold, exist_ok=True)
        print("HI", top_inds.shape, random_indices.shape, feature_ids.shape)
        print(feature_ids.shape)
        print(highest_activation.shape)
        print("sig")
        generate_images(
            feature_ids, 
            feature_hit_count,
            highest_activation,
            imagenet_data,
            imagenet_index_to_name,
            pretrained_layers,
            fold,
            border_color=border_color,
            header=name.replace("_"," "),
        )

    
    print("TODO SPLIT:")


