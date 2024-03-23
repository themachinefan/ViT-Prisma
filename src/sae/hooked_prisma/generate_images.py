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
from sae.language.sae_group import SAEGroup
from vit_prisma.models.base_vit import HookedViT

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
    num_autoencoders,k, num_features 148,  (where 148 = batchid, patchid, true val, 145 decomposed vals)  
    '''
    # Get the post activations from the clean run
    _, cache = model.run_with_cache(images)
    # sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid = sparse_autoencoder(
    #     cache[sparse_autoencoder.cfg.hook_point]
    # )



    decomposition,labels = cache.get_full_resid_decomposition(return_labels=True, expand_neurons=False) #Parts Batch Patch Dim
    
    #indices = [i for i, item in enumerate(labels) if item in items_to_find]
    largest_activations = []
    smallest_activations = []

    for layer_i, (cur_feature_ids, autoencoder) in enumerate(zip(feature_ids, autoencoders)):
        relevant_inputs = ['embed', 'pos_embed'] 
        for li in range(layer_i):
            relevant_inputs =  relevant_inputs + [f"{li}_mlp_out"] + [f"L{li}H{hi}" for hi in range(model.cfg.n_heads)]
        indices = [i for i, item in enumerate(labels) if item in relevant_inputs]
        inp = decomposition[indices]
        post_reshaped = einops.rearrange( inp, "decom batch seq d_mlp -> decom (batch seq) d_mlp")
        seq_len = cache[autoencoder.cfg.hook_point].shape[1]
        sae_in =  post_reshaped #- autoencoder.b_dec  NOTE we want to decompose so we do this below (won't change which has largest impact per feature)
        acts_decomp = einops.einsum(
                sae_in,
                autoencoder.W_enc[:, cur_feature_ids], #dim numfeatures
                "decomp ... d_in, d_in n -> decomp ... n",
            )
        acts = acts_decomp.sum(0) 
        
        bias = einops.einsum(
                autoencoder.b_dec,
                autoencoder.W_enc[:, cur_feature_ids],
                "... d_in, d_in n -> ... n",
            )
        acts = acts - bias
        

        
        extremes = []
        for is_largest in [True,False]:
            top_acts_values, top_acts_indices = acts.topk(k, dim=0, largest=is_largest)
            expanded_indices = top_acts_indices.unsqueeze(0).expand(acts_decomp.shape[0], *top_acts_indices.shape)
            top_acts_values_decomp = torch.gather(acts_decomp, 1, expanded_indices)

           # assert torch.allclose(top_acts_values , top_acts_values_decomp.sum(0)) # check only works without bias above
            top_acts_values_decomp = einops.rearrange(top_acts_values_decomp, "a b c -> b c a")

            top_acts_values_decomp = torch.nn.functional.pad(top_acts_values_decomp, (0,145-top_acts_values_decomp.shape[-1]), "constant", 0)
            top_acts_batch = top_acts_indices // seq_len
            top_acts_seq = top_acts_indices % seq_len
 
            largest = torch.concatenate([top_acts_batch.unsqueeze(-1), top_acts_seq.unsqueeze(-1), top_acts_values.unsqueeze(-1), top_acts_values_decomp], dim=-1)
            extremes.append(largest)
        largest_activations.append(extremes[0])
        smallest_activations.append(extremes[1])


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

    n, m, p = len(autoencoders), feature_ids.shape[1], 148
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
        map_output_folder,
        header="",
        border_color=(0,0,0),
        patch_size=32,
):
    BID = 0
    PID = 1
    VAL = 2
    DECOMP_START = 3
    for i in range(features_ids.shape[0]):
        autoencoder_id = autoencoder_layer_names[i] 
        for ii, feature_id in enumerate(features_ids[i]):

            if os.path.exists(os.path.join(output_folder,f'layer_{autoencoder_id}_feature_{feature_id}.png')):
                print("HACK")
                continue
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


            source = torch.zeros((145,)).to(highest_activations.device)
            for grid_i, (image_tensor, label, iii) in enumerate(zip(images, gt_labels, list(range(highest_activations.shape[2])))):

                source = source + highest_activations[i, ii, iii, DECOMP_START:]

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

   

                            
            from io import BytesIO

            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Load the buffer as an image using OpenCV
            img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            # Specify the border size and color
            border_size = 10

            # Add the border to the image
            bordered_img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=border_color)
            bordered_img[0,0,:] = 0
            cv2.imwrite(os.path.join(output_folder,f'layer_{autoencoder_id}_feature_{feature_id}.png'), bordered_img)

            plt.close()


   
            source = source.detach().cpu().numpy()

            source_head = np.zeros((13,)) 
            source_head[0:2] = source[0:2]

            source = np.concatenate([ source_head[None], np.reshape(source[2:], (-1,13)) ], axis=0)

            #gross clean up
            if border_color[1] == 0:
                source = -source

            n,m = source.shape    
            fig, ax = plt.subplots()
            fig.suptitle(f'Important sources\nLayer {autoencoder_id}, Feature {feature_id}, hit count: {features_hit_count[i,ii]}')





            cax = ax.imshow(source, cmap='viridis', interpolation='nearest')
            plt.colorbar(cax)

            ax.set_xticks(np.arange(0, m), minor=False)
            ax.set_yticks(np.arange(0, n), minor=False)

            
            ax.set_xticklabels(["MLP"] + [f"H{hi}" for hi in range(12)])
            ax.set_yticklabels(["embed/pos"] + [f"L{li}" for li in range(11)])

            ax.set_xticks(np.arange(-0.5, m - 0.5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n - 0.5, 1), minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

            ax.tick_params(which="major", bottom=True, left=True, labelbottom=True, labelleft=True)

            # Save

            plt.savefig(os.path.join(map_output_folder,f'layer_{autoencoder_id}_feature_{feature_id}.png'))            
            plt.close()
          #  fig.savefig(os.path.join(output_folder,f'layer_{autoencoder_id}_feature_{feature_id}.png'), dpi=300)

if __name__ == "__main__":
    use_cache = True
    batch_size = 64
    num_workers = 0
    device = "cuda" #not actually set up for cpu atm :P 
    checkpoint_path= "F:/ViT-Prisma_fork/data/vision_sae_checkpoints"
    imagenet_path = "F:/prisma_data/imagenet-object-localization-challenge"
    pretrained_path="F:/ViT-Prisma_fork/data/vision_sae_checkpoints/organized"
    output_path = "F:/ViT-Prisma_fork/data/feature_images"
    output_cache = os.path.join(output_path, "cache")
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(output_cache,exist_ok=True)


    pretrained_model_name = "vit_base_patch32_224"
    model = HookedViT.from_pretrained(pretrained_model_name)
    model.to(device)



    saes = []
    pretrained_layers = []
    for i in range(12):
        pretrained_layers.append(i)
        sae_group = SAEGroup.load_from_pretrained(os.path.join(pretrained_path, f"layer_{i}.pt"))

        saes.append(sae_group.autoencoders[0])

 



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
        hit_count = count_activated(
            dataloader,
            model,
            saes,
            device=device)
        torch.save(hit_count, hit_count_path)
    
    top_amount = 20
    random_amount = 20
    top_vals, top_inds = hit_count.topk(top_amount, dim=1)
    




    num_images = 25
    print("TODO SAVE CACHE (could include loading hitcount)")
    highest_activating_path = os.path.join(output_cache, "highest_activating.pt" )

    if os.path.exists(highest_activating_path) and use_cache:
        highest_activation, lowest_activation, all_indices, random_indices, random_values = torch.load(highest_activating_path)
    else:

        torch.manual_seed(0)



        random_indices = []
        random_values = []
        for i in range(hit_count.shape[0]):
            cur_hit_counts = hit_count[i]
            notable_cur_hit_counts_inds = torch.where(cur_hit_counts>200)[0]
            inds = notable_cur_hit_counts_inds[torch.randperm(notable_cur_hit_counts_inds.size(0))[:random_amount]]
            random_values.append(cur_hit_counts[inds])
            random_indices.append(inds)

        random_values = torch.stack(random_values)
        random_indices = torch.stack(random_indices)

       
        all_indices = torch.cat([top_inds, random_indices], dim=1)


        highest_activation, lowest_activation = find_highest_activating(dataloader, model, saes, all_indices,k=num_images, device=device )
        torch.save([highest_activation,lowest_activation, all_indices, random_indices, random_values], highest_activating_path)

    ha_top, ha_rand = highest_activation[:, :top_amount], highest_activation[:, top_amount:]
    la_top, la_rand = lowest_activation[:, :top_amount], lowest_activation[:, top_amount:]

    for name, feature_ids, feature_hit_count, highest_activation, border_color in zip(["top_feature_high_activations","top_feature_low_activations","random_feature_high_activations","random_feature_low_activations"],
                                                                        [top_inds, top_inds, random_indices, random_indices],
                                                                        [top_vals,top_vals,random_values,random_values],
                                                                        [ha_top, la_top, ha_rand, la_rand],
                                                                   #     [(0.0,1.0,0.0), (0.,0.0,1.), (0.,1.,0.), (0.,0.,1.)]
                                                                        [(0,200,0), (0,0,200), (0,200,0), (0,0,200)]

                                                                        ):
        fold = os.path.join(output_path, name)
        os.makedirs(fold, exist_ok=True)
        map_output_folder = os.path.join(output_path, name + "_sourcemap")
        os.makedirs(map_output_folder,exist_ok=True)
        generate_images(
            feature_ids, 
            feature_hit_count,
            highest_activation,
            imagenet_data,
            imagenet_index_to_name,
            pretrained_layers,
            fold,
            map_output_folder, 
            border_color=border_color,
            header=name.replace("_"," "),
        )

    


