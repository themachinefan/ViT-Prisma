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
import torch
import webbrowser
from pyvis.network import Network
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import heapq

from sparse_circuit.demo import setup_saes_and_model, get_imagenet_val_dataset, get_imagenet_val_dataset_visualize
def find_threshold_for_top_k(A, k):
    # Find the kth largest value, which is the threshold for the top k elements
    if k > len(A):
        return None 
    top_k = heapq.nlargest(k, A)
    n = top_k[-1]
    return n
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

    print(sparse_autoencoder.b_enc.shape, sparse_autoencoder.W_enc.shape)
    print(interesting_features_indices)
   # print(sparse_autoencoder.b)
    #sparse_autoencoder.b_enc =  sparse_autoencoder.b_enc.to('cpu')
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

    post_reshaped = einops.rearrange(sparse_autoencoder.run_time_activation_norm_fn_in(cache[sparse_autoencoder.cfg.hook_point]), "batch seq d_mlp -> (batch seq) d_mlp")
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

    post_reshaped = sparse_autoencoder.run_time_activation_norm_fn_in(cache[sparse_autoencoder.cfg.hook_point])
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
def visualize_top_activating_images(save_folder, root_name, model, sae, top_activations_per_feature,feature_ids, attrib_values, dataset, dataset_visualize, device, patch_size=32, output_image_format="jpg"):
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
            axs[row, col].axis('off')  

        plt.tight_layout()

        plt.savefig(os.path.join(save_folder, f"{root_name}_{feature_id}.{output_image_format}"), format=output_image_format)
        plt.close()










if __name__ == "__main__":

    ### SETUP #########################################
    # load the output from demo.py 
    parsed_nodes_path =  r"F:\ViT-Prisma_fork\data\circuit_output\testing_stuff_2_nodes_parsed.pt"
    edges_path = r"F:\ViT-Prisma_fork\data\circuit_output\testing_stuff_2_edges.pt"
    nodes_indices_loaded, nodes_values_loaded = torch.load(parsed_nodes_path)
    edges_loaded= torch.load(edges_path)

    print(nodes_indices_loaded, nodes_values_loaded)
    # load the imagenet dataset to be used for getting maxmimal activating images for each feature 
    imagenet_dataset_path = r"F:/prisma_data/imagenet-object-localization-challenge"
    output_folder = r"F:\ViT-Prisma_fork\data\circuit_output\testing_stuff_output"

    device = "cuda"
    output_image_format = "jpg"
    batch_size = 32
    do_generate_images = False # set to false if already done
    auto_open_browse_on_completion = True
    top_k_edges = 50 # only keep this many edges in total. (can be none)
    edge_threshold = None # only keep edges above this threshold (can be none)

    #autofind what layers were used (assuming only using the residual stream like the rest of the code)
    only_these_layers = []
    for hook_point in nodes_values_loaded.keys():
        only_these_layers.append(int(hook_point.split('.')[1]))

    model, saes, model_name = setup_saes_and_model(device, debug=False, only_these_layers=only_these_layers)


    dataset = get_imagenet_val_dataset(imagenet_dataset_path)
    visualize_dataset = get_imagenet_val_dataset_visualize(imagenet_dataset_path)
    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)



    ##### SETUP END #########################################################




    ###### GENERATE IMAGES #############################################

    im_folder = os.path.join(output_folder, 'images')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(im_folder, exist_ok=True)


    if do_generate_images:
        # The first step is to get highly activating images for each node as a crude way of autointerpreting
        ### Get images for each feature
        for hook_point in nodes_indices_loaded.keys():
            sae = saes[hook_point]

            feature_ids = nodes_indices_loaded[hook_point]
            feature_vals = nodes_values_loaded[hook_point]

            max_feature = sae.d_sae 
        
            # the max_feature represents sae error, so it has no associated autointerp, so we represent it with black image 
            pruned_feature_ids = []
            pruned_feature_vals = []
            error_features = []
            error_vals = []
            for i, v in zip(feature_ids, feature_vals):
                if i != max_feature:
                    pruned_feature_ids.append(i)
                    pruned_feature_vals.append(v)
                else:
                    error_features.append(i)
                    error_vals.append(v)
            if pruned_feature_ids:
                top_activations_per_feature = find_top_activations(
                    dataloader, model, sae,
                    pruned_feature_ids, [False]*len(pruned_feature_ids), batch_size=batch_size, top_k=16, max_samples=50_000,
                )

                root_name = re.sub(r'\b\d+\b', lambda x: f"{int(x.group()):02d}", hook_point)
                visualize_top_activating_images(im_folder, root_name, model, sae, top_activations_per_feature, pruned_feature_ids, pruned_feature_vals, dataset, visualize_dataset, device)

            if error_features:
                for e in error_features:
                    # make a dummy image!

                    # Create a black image of size 100x100
                    black_image = Image.new("RGB", (100, 100), (0, 0, 0))

                    black_image_path =  os.path.join(im_folder, f"{root_name}_{e}.{output_image_format}")

                    # add text
                                        
                    draw = ImageDraw.Draw(black_image)
                    image_width, image_height = black_image.size
                    lines = ["SAE", "error", "term"]
                    font_size = 100
                    while font_size > 0:
                        try:
                            font = ImageFont.truetype("arial.ttf", font_size)
                        except IOError:
                            # Fallback to default font if TrueType font is not available
                            font = ImageFont.load_default()
                            break
                        total_height = sum([draw.textsize(line, font=font)[1] for line in lines])
                        max_line_width = max([draw.textsize(line, font=font)[0] for line in lines])
                        if total_height <= image_height and max_line_width <= image_width:
                            break
                        font_size -= 1
                    else:
                        font = ImageFont.load_default()

                    y_text = (image_height - total_height) / 2

                    for line in lines:

                        text_width, text_height = draw.textsize(line, font=font)
                        x_text = (image_width - text_width) / 2
                        draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255))
                        y_text += text_height                          
                black_image.save(black_image_path)


    ###### END GENERATE IMAGES  #############################################
                
    ###### CREATE GRAPH #####################################################
    # parse the edges
                
    top_k_edges = 2 # only keep this many edges in total. 
    edge_threshold = None # only keep edges above this threshold
    all_edge_vals = []
    for src_hook_point in edges_loaded.keys():
        for dst_hook_point, cur_edges in edges_loaded[src_hook_point].items():
            if dst_hook_point in nodes_indices_loaded.keys():
                for dst_ind in nodes_indices_loaded[dst_hook_point]:
                    if src_hook_point in nodes_indices_loaded.keys():
                        values = cur_edges[dst_ind][nodes_indices_loaded[src_hook_point]].tolist()
                        for val, src_ind in zip(values,nodes_indices_loaded[src_hook_point] ):
                            all_edge_vals.append(np.abs(val))

    edges = {}
    top_k_edges_thresh=None
    if top_k_edges is not None:
        top_k_edges_thresh = find_threshold_for_top_k(all_edge_vals, top_k_edges)
    for src_hook_point in edges_loaded.keys():
        edges[src_hook_point] = {}
        for dst_hook_point, cur_edges in edges_loaded[src_hook_point].items():
            
            edges[src_hook_point][dst_hook_point] = {}
            if dst_hook_point in nodes_indices_loaded.keys():
                for dst_ind in nodes_indices_loaded[dst_hook_point]:
                    if src_hook_point in nodes_indices_loaded.keys():
                        
                        values = cur_edges[dst_ind][nodes_indices_loaded[src_hook_point]].tolist()

                        #TODO sparse tensor instead 
                        for val, src_ind in zip(values,nodes_indices_loaded[src_hook_point] ):
                            if edge_threshold is not None and np.abs(val)<edge_threshold:
                                continue 
                            if top_k_edges_thresh is not None and np.abs(val) < top_k_edges_thresh:
                                continue 
                            if src_ind not in  edges[src_hook_point][dst_hook_point].keys():
                                edges[src_hook_point][dst_hook_point][src_ind] = {} 
                            edges[src_hook_point][dst_hook_point][src_ind][dst_ind] = val 

    print(edges)    
    #TODO parse edges!

    # create the graph object
    

        

    G = nx.Graph()

    # add nodes to graph
    node_id = 0 
    node_ids = {}

    hook_points = list(nodes_indices_loaded.keys())

    # Desired node image size
    node_size = 200  # Adjust this value to make images larger or smaller, make sure to adjust spacing as well
    row_spacing = node_size + 600 
    col_spacing = node_size+ 300  # adjust as needed

    for row_index, hook_point in enumerate(hook_points):
        node_ids[hook_point] = {}
        inds = nodes_indices_loaded[hook_point]
        vals = nodes_values_loaded[hook_point]
        for col_index, (ind, val) in enumerate(zip(inds, vals)):
            hook_blah = re.sub(r'\b\d+\b', lambda x: f"{int(x.group()):02d}", hook_point)
            root_name = f"{hook_blah}_{ind}"
            image_path = os.path.join(im_folder, f"{root_name}.{output_image_format}")
            
            # Calculate positions
            x = col_index * col_spacing
            y = row_index * row_spacing  
            
            G.add_node(
                node_id,
                title=f"{root_name} | Value: {val}",
                image=f"{root_name}.jpg",
                shape="image",
                size=node_size,  
                x=x,
                y=y,
            )
            
            node_ids[hook_point][ind] = node_id
            node_id += 1

    # add edges to graph (unchanged)
    for src_hook_point, inner in edges.items():
        for dst_hook_point in edges[src_hook_point].keys():
            for src_feature in edges[src_hook_point][dst_hook_point].keys():
                for dst_feature, edge_val in edges[src_hook_point][dst_hook_point][src_feature].items():
                    color = "blue" if edge_val > 0 else "red"
                    G.add_edge(
                        node_ids[src_hook_point][src_feature],
                        node_ids[dst_hook_point][dst_feature],
                        color=color,
                        title=f"Value: {edge_val}"
                    )

    # create network
    net = Network(width="100%", height="800px")
    net.from_nx(G)

    # Set positions and fix them
    for node in net.nodes:
        node_id = node['id']
        x = G.nodes[node_id]['x']
        y = G.nodes[node_id]['y']
        node['x'] = x
        node['y'] = y
        node['fixed'] = {'x': True, 'y': True}
        node['physics'] = False  # Ensure physics is disabled for individual nodes

    # Disable physics for the entire network
    net.toggle_physics(False)

    # save and display network
    output_file = os.path.join(im_folder, "graph_with_values.html")  # must be saved in same folder as images
    net.write_html(output_file)



    if auto_open_browse_on_completion:
        webbrowser.open(output_file) 

        ###### END CREATE GRAPH #################################################

