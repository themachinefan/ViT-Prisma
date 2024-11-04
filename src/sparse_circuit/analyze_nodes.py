
import torch 

from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
from typing import List, Dict, Tuple
import einops

from typing import List 

import re 
from PIL import Image

from sparse_circuit.nodes_only_demo import setup, get_imagenet_val_dataset, get_imagenet_val_dataset_visualize

@torch.no_grad()
def compute_feature_activations(
    images: torch.Tensor,
    model: torch.nn.Module,
    sparse_autoencoder: torch.nn.Module,
    encoder_weights: torch.Tensor,
    encoder_biases: torch.Tensor,
    feature_ids: List[int],
    is_cls_list: List[bool],
    top_k: int = 10
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute the highest activating tokens for given features in a batch of images.
    
    Args:
        images: Input images
        model: The main model
        sparse_autoencoder: The sparse autoencoder
        encoder_weights: Encoder weights for selected features
        encoder_biases: Encoder biases for selected features
        feature_ids: List of feature IDs to analyze
        feature_categories: Categories of the features
        top_k: Number of top activations to return per feature

    Returns:
        Dictionary mapping feature IDs to tuples of (top_indices, top_values)
    """
    _, cache = model.run_with_cache(images, names_filter=[sparse_autoencoder.cfg.hook_point])
    
    layer_activations = cache[sparse_autoencoder.cfg.hook_point]
    batch_size, seq_len, _ = layer_activations.shape
    flattened_activations = einops.rearrange(layer_activations, "batch seq d_mlp -> (batch seq) d_mlp")
    
    sae_input = flattened_activations - sparse_autoencoder.b_dec
    feature_activations = einops.einsum(sae_input, encoder_weights, "... d_in, d_in n -> ... n") + encoder_biases
    feature_activations = torch.nn.functional.relu(feature_activations)
    
    reshaped_activations = einops.rearrange(feature_activations, "(batch seq) d_in -> batch seq d_in", batch=batch_size, seq=seq_len)
    cls_token_activations = reshaped_activations[:, 0, :]
    mean_image_activations = reshaped_activations.mean(1)

    top_activations = {}
    for i, (feature_id, is_cls) in enumerate(zip(feature_ids, is_cls_list)):
        if is_cls:
            top_values, top_indices = cls_token_activations[:, i].topk(top_k)
        else:
            top_values, top_indices = mean_image_activations[:, i].topk(top_k)
        top_activations[feature_id] = (top_indices, top_values)
    
    return top_activations

def find_top_activations(
    val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    sparse_autoencoder: torch.nn.Module,
    interesting_features_indices: List[int],
    is_cls_list: List[bool],
    top_k: int = 16,
    max_samples= 50_000,
    batch_size = 54, 
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Find the top activations for interesting features across the validation dataset.

    Args:
        val_dataloader: Validation data loader
        model: The main model
        sparse_autoencoder: The sparse autoencoder
        interesting_features_indices: Indices of interesting features
        interesting_features_category: Categories of interesting features

    Returns:
        Dictionary mapping feature IDs to tuples of (top_values, top_indices)
    """
    device = next(model.parameters()).device
    top_activations = {i: (None, None) for i in interesting_features_indices}

    encoder_biases = sparse_autoencoder.b_enc[interesting_features_indices]
    encoder_weights = sparse_autoencoder.W_enc[:, interesting_features_indices]

    processed_samples = 0
    for batch_images, _, batch_indices in tqdm(val_dataloader, total=max_samples // batch_size):
        batch_images = batch_images.to(device)
        batch_indices = batch_indices.to(device)
        batch_size = batch_images.shape[0]

        batch_activations = compute_feature_activations(
            batch_images, model, sparse_autoencoder, encoder_weights, encoder_biases,
            interesting_features_indices, is_cls_list, top_k
        )

        for feature_id in interesting_features_indices:
            new_indices, new_values = batch_activations[feature_id]
            new_indices = batch_indices[new_indices]
            
            if top_activations[feature_id][0] is None:
                top_activations[feature_id] = (new_values, new_indices)
            else:
                combined_values = torch.cat((top_activations[feature_id][0], new_values))
                combined_indices = torch.cat((top_activations[feature_id][1], new_indices))
                _, top_k_indices = torch.topk(combined_values, top_k)
                top_activations[feature_id] = (combined_values[top_k_indices], combined_indices[top_k_indices])

        processed_samples += batch_size
        if processed_samples >= max_samples:
            break

    return {i: (values.detach().cpu(), indices.detach().cpu()) 
            for i, (values, indices) in top_activations.items()}

torch.no_grad()
def get_heatmap(
          image,
          model,
          sparse_autoencoder,
          feature_id,
          device,
): 
    image = image.to(device)
    _, cache = model.run_with_cache(image.unsqueeze(0), names_filter=[sparse_autoencoder.cfg.hook_point])

    post_reshaped = einops.rearrange(sae.run_time_activation_norm_fn_in(cache[sparse_autoencoder.cfg.hook_point]), "batch seq d_mlp -> (batch seq) d_mlp")
    # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    sae_in =  post_reshaped - sparse_autoencoder.b_dec # Remove decoder bias as per Anthropic

    acts = einops.einsum(
            sae_in,
            sparse_autoencoder.W_enc[:, feature_id],
            "x d_in, d_in -> x",
        )
    return acts 
@torch.no_grad()
def get_heatmap_batch(
          images,
          model,
          sparse_autoencoder,
          feature_id,
          device,
): 
    image = torch.stack(images, dim=0).to(device)
    _, cache = model.run_with_cache(image, names_filter=[sparse_autoencoder.cfg.hook_point])

    post_reshaped = sae.run_time_activation_norm_fn_in(cache[sparse_autoencoder.cfg.hook_point])
    # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
    # This code is copied from the first part of the 'forward' method of the AutoEncoder class
    sae_in =  post_reshaped - sparse_autoencoder.b_dec # Remove decoder bias as per Anthropic

    acts = einops.einsum(
            sae_in,
            sparse_autoencoder.W_enc[:, feature_id],
            "b x d_in, d_in -> b x",
        )
    return acts 
     
def image_patch_heatmap(activation_values,image_size=224, pixel_num=14):
    activation_values = activation_values.detach().cpu().numpy()
    activation_values = activation_values[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)

    # Create a heatmap overlay
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num

    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]

    return heatmap



    # Removing axes
#TODO clean this up and do a batch version!
def visualize_top_activating_images(save_folder, root_name, model, sae, top_activations_per_feature,feature_ids, attrib_values, dataset, dataset_visualize, device, patch_size=32):
    importance = 0
    for feature_id, attrib_value in tqdm(zip(feature_ids, attrib_values)):
        importance += 1
        max_vals, max_inds = top_activations_per_feature[feature_id]
        images = []
        model_images = []
        for bid in max_inds:

            image, _, image_ind = dataset_visualize[bid]

            assert image_ind.item() == bid
            images.append(image)

            model_image, _, _ = dataset[bid]
            model_images.append(model_image)
        
        grid_size = int(np.ceil(np.sqrt(len(images))))
        fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
        name=  f"Feature: {feature_id} Node val: {attrib_value}"
        fig.suptitle(name)#, y=0.95)
        for ax in axs.flatten():
            ax.axis('off')
        complete_bid = []
        heatmaps = get_heatmap_batch(model_images,model,sae, feature_id,device )

        for i, (image_tensor, val, bid, heatmap) in enumerate(zip(images, max_vals,max_inds,heatmaps )):
            if bid in complete_bid:
                continue 
            complete_bid.append(bid)


            row = i // grid_size
            col = i % grid_size
            heatmap = image_patch_heatmap(heatmap, pixel_num=224//patch_size)

            display = image_tensor.numpy().transpose(1, 2, 0)


            axs[row, col].imshow(display)
            axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)  # Overlaying the heatmap
            axs[row, col].set_title(f"{val.item():0.03f}")  
            axs[row, col].axis('off')  

        plt.tight_layout()

        plt.savefig(os.path.join(save_folder, f"{root_name}_{importance:03d}_{attrib_value:.2f}.jpg"), format="jpg")
        plt.close()



if __name__ == "__main__":


    # since I just have nodes AND since I've just naively run on the whole dataset I'm going to do a simple outlier search

    device = "cuda"
    batch_size = 64
    residual_error_index = 49152
    imagenet_dataset_path = r"F:/prisma_data/imagenet-object-localization-challenge"

    output_folder = r"F:\ViT-Prisma_fork\data\circuit_output"
    saved_nodes_path = os.path.join(output_folder, 'final_250_cats_only.pt')
    im_folder = os.path.join(output_folder, 'images')
    os.makedirs(im_folder, exist_ok=True)

    final_nodes_loaded = torch.load(saved_nodes_path)


    
    model, saes, model_name = setup(device)
    
        #clip_text_stuff = ClipTextStuff(model_name, device)





    dataset = get_imagenet_val_dataset(imagenet_dataset_path)
    visualize_dataset = get_imagenet_val_dataset_visualize(imagenet_dataset_path)


    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


    # start by visualizing 
    best_nodes = {}
    for hook_point, results in final_nodes_loaded.items():


        mean = torch.mean(results)
        std = torch.std(results)

        # Define the number of standard deviations
        num_std = 3  # For values greater than one standard deviation above the mean
        max_amount = 20 # just to keep the analysis smaller... 

        # Find indices where A > mean + num_std * std
        indices = torch.nonzero(results > mean + num_std * std).squeeze()
        values = results[indices]

        if max_amount is not None:
            if len(values)> max_amount:
                og_amount = len(values)   
                values, top_k_indices = torch.topk(values, max_amount, largest=True)
                indices = indices[top_k_indices]
                print("removed", og_amount-max_amount, 'to save time')

        values, sorted_indices = torch.sort(values, descending=True)
        indices = indices[sorted_indices]


        if residual_error_index in indices:
            removed_position = (indices == residual_error_index).nonzero(as_tuple=True)[0].item()
            
            removed_value = values[removed_position]
            
            values = torch.cat((values[:removed_position], values[removed_position+1:]))
            indices = torch.cat((indices[:removed_position], indices[removed_position+1:]))
        else:
            removed_value = None
            removed_position = None 

        best_nodes[hook_point] = (values, indices, removed_value, removed_position)
        display = False 
        if display:
            results_cpu = results.cpu().numpy()
            indices_cpu = indices.cpu().numpy()
            values_cpu = values.cpu().numpy()
            n = len(results_cpu)
            # Create an array of indices for all elements in A
            x = np.arange(n)

            # Create a boolean mask for selected indices
            selected_mask = np.zeros(n, dtype=bool)
            selected_mask[indices_cpu] = True

            # **Step 3: Plot the Data**

            plt.figure(figsize=(12, 6))

            # Plot all values in A
            plt.scatter(x[~selected_mask], results_cpu[~selected_mask], color='blue', label='Other Values')

            # Highlight the selected values
            plt.scatter(x[selected_mask], results_cpu[selected_mask], color='red', label='Selected Values')

            # **Optional: Add Labels and Title**
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'{hook_point} Selected Values Highlighted')
            plt.legend()
            plt.grid(True)
            plt.show()

    

    



    hook_points_ordered = list(best_nodes.keys())[::-1] # I just want to start with later layers
    for hook_point in hook_points_ordered:
        values, feature_ids,  res_value, res_position = best_nodes[hook_point]
        sae = saes[hook_point]

        feature_ids = feature_ids.tolist()
        top_activations_per_feature = find_top_activations(
            dataloader, model, sae,
            feature_ids, [False]*len(feature_ids), batch_size=batch_size, top_k=16, max_samples=50_000,
        )

        root_name = re.sub(r'\b\d+\b', lambda x: f"{int(x.group()):02d}", hook_point)
        visualize_top_activating_images(im_folder, root_name, model, sae, top_activations_per_feature, feature_ids, values.tolist(), dataset, visualize_dataset, device)

        if res_value is not None:
            # make a dummy image!

            # Create a black image of size 100x100
            black_image = Image.new("RGB", (100, 100), (0, 0, 0))

            # Save the black image
            black_image_path =  os.path.join(im_folder, f"{root_name}_{res_position:03d}_{res_value:.2f}_RESIDUAL_ERROR.png")
                                             
            black_image.save(black_image_path)
           