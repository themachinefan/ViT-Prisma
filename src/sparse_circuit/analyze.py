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
from copy import copy 
from sparse_circuit.demo import setup_saes_and_model, get_imagenet_val_dataset, get_imagenet_val_dataset_visualize
from vit_prisma.sae.sae import SparseAutoencoder
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
    sparse_autoencoders: torch.nn.Module,
    feature_ids: List[int],
    is_cls_list: List[bool],
    top_k: int = 10
) -> Dict[str,Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Compute the highest activating tokens for given features in a batch of images.
    
    Args:
        images: Input images
        model: The main model
        sparse_autoencoders: The sparse autoencoders (dictionary: hook_point -> sparse autoencoder)
        feature_ids: Indices of features (dictionary: hook_point -> features)
        is_cls_list: tells which indices are cls or not (dictionary: hook_point -> list of bool)
        top_k: Number of top activations to return per feature

    Returns:
        Dictionary of dictionary mapping hookpoint to feature IDs to tuples of (top_indices, top_values)
    """

    _, cache = model.run_with_cache(images, names_filter=list(sparse_autoencoders.keys()))
    top_activations = {}
    for hook_point in sparse_autoencoders.keys():
        top_activations[hook_point] = {}

        cur_feature_ids = feature_ids[hook_point]
        if len(cur_feature_ids) == 0:
            top_activations[hook_point][feature_id] = (None, None)
            continue 
        sparse_autoencoder = sparse_autoencoders[hook_point]

        cur_is_cls_list = is_cls_list[hook_point]
        encoder_biases = sparse_autoencoder.b_enc[cur_feature_ids]
        encoder_weights = sparse_autoencoder.W_enc[:, cur_feature_ids]

        layer_activations = cache[hook_point]
        batch_size, seq_len, _ = layer_activations.shape
        flattened_activations = einops.rearrange(layer_activations, "batch seq d_mlp -> (batch seq) d_mlp")
        
        sae_input = flattened_activations - sparse_autoencoder.b_dec
        feature_activations = einops.einsum(sae_input, encoder_weights, "... d_in, d_in n -> ... n") + encoder_biases
        feature_activations = torch.nn.functional.relu(feature_activations)
        
        reshaped_activations = einops.rearrange(feature_activations, "(batch seq) d_in -> batch seq d_in", batch=batch_size, seq=seq_len)
        cls_token_activations = reshaped_activations[:, 0, :]
        mean_image_activations = reshaped_activations.mean(1)

        for i, (feature_id, is_cls) in enumerate(zip(cur_feature_ids, cur_is_cls_list)):
            if is_cls:
                top_values, top_indices = cls_token_activations[:, i].topk(top_k)
            else:
                top_values, top_indices = mean_image_activations[:, i].topk(top_k)
            top_activations[hook_point][feature_id] = (top_indices, top_values)
        
    return top_activations

def find_top_activations(
    val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    sparse_autoencoders: Dict[str, SparseAutoencoder],
    interesting_features_indices: Dict[str, List[int]],
    is_cls_list:  Dict[str,List[bool]],
    top_k: int = 16,
    max_samples= 50_000,
    batch_size = 54, 
) -> Dict[str,Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Find the top activations for interesting features across the validation dataset.

    Args:
        val_dataloader: Validation data loader
        model: The main model
        sparse_autoencoders: The sparse autoencoders (dictionary: hook_point -> sparse autoencoder)
        interesting_features_indices: Indices of interesting features (dictionary: hook_point -> features)
        is_cls_list: tells which indices are cls or not (dictionary: hook_point -> list of bool)

    Returns:
        Dictionary of dictionary mapping hookpoints to feature IDs to tuples of (top_values, top_indices)
    """
    device = next(model.parameters()).device
    top_activations= {}
    for hp, inds in interesting_features_indices.items():
        top_activations[hp] = {i: (None, None) for i in inds}

   # print(sparse_autoencoder.b)
    #sparse_autoencoder.b_enc =  sparse_autoencoder.b_enc.to('cpu')

    processed_samples = 0
    for batch_images, _, batch_indices in tqdm(val_dataloader, total=max_samples // batch_size):
        batch_images = batch_images.to(device)
        batch_indices = batch_indices.to(device)
        batch_size = batch_images.shape[0]

        batch_activations = compute_feature_activations(
            batch_images, model, sparse_autoencoders,
            interesting_features_indices, is_cls_list, top_k
        )

        for hp in interesting_features_indices.keys():
            for feature_id in interesting_features_indices[hp]:
                new_indices, new_values = batch_activations[hp][feature_id]

                new_indices = batch_indices[new_indices]
                
                if top_activations[hp][feature_id][0] is None:
                    top_activations[hp][feature_id] = (new_values, new_indices)
                else:
                    combined_values = torch.cat((top_activations[hp][feature_id][0], new_values))
                    combined_indices = torch.cat((top_activations[hp][feature_id][1], new_indices))
                    _, top_k_indices = torch.topk(combined_values, top_k)
                    top_activations[hp][feature_id] = (combined_values[top_k_indices], combined_indices[top_k_indices])

        processed_samples += batch_size
        if processed_samples >= max_samples:
            break

    return {
            outer_key: {
                inner_key: (values.detach().cpu(), indices.detach().cpu())
                for inner_key, (values, indices) in inner_dict.items()
            }
            for outer_key, inner_dict in top_activations.items()
        }


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
def visualize_top_activating_images(save_folder, root_name, model, sae, top_activations_per_feature,feature_ids, dataset, dataset_visualize, device, patch_size=32, output_image_format="jpg"):
    importance = 0
    for feature_id in tqdm(feature_ids):
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
    input_name = "testing_stuff_3_saes"
    parsed_nodes_path =  fr"F:\ViT-Prisma_fork\data\circuit_output\{input_name}_nodes_parsed.pt"
    edges_path = fr"F:\ViT-Prisma_fork\data\circuit_output\{input_name}_edges.pt"
    output_folder = fr"F:\ViT-Prisma_fork\data\circuit_output\{input_name}_output_all_nodes"

    nodes_indices_loaded, nodes_values_loaded = torch.load(parsed_nodes_path)
    edges_loaded= torch.load(edges_path)

    print(nodes_indices_loaded, nodes_values_loaded)
    # load the imagenet dataset to be used for getting maxmimal activating images for each feature 
    imagenet_dataset_path = r"F:/prisma_data/imagenet-object-localization-challenge"

    device = "cuda"
    output_image_format = "jpg"
    batch_size = 32
    do_generate_images = True # set to false if already done
    auto_open_browse_on_completion = True
    top_k_edges = 50 # only keep this many edges in total. (can be none)
    edge_threshold = 1e-4 #None # only keep edges above this threshold (can be none)
    use_neurons = False # was the circuit computed with neurons? 
    add_nodes_from_edges = True # if a node has an prominent edge, add it.

    #autofind what layers were used (assuming only using the residual stream like the rest of the code)
    only_these_layers = []
    for hook_point in nodes_values_loaded.keys():
        only_these_layers.append(int(hook_point.split('.')[1]))

    model, saes, model_name = setup_saes_and_model(device, debug=False, only_these_layers=only_these_layers, use_neurons=use_neurons)


    dataset = get_imagenet_val_dataset(imagenet_dataset_path)
    visualize_dataset = get_imagenet_val_dataset_visualize(imagenet_dataset_path)
    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)



    ##### SETUP END #########################################################

  # add nodes from edges
    #print(nodes_indices_loaded)             
    #TODO this looks back once, could do more 
    if add_nodes_from_edges:
        for src_hook_point in edges_loaded.keys():
            initial_indices = copy(nodes_indices_loaded[src_hook_point])
            for dst_hook_point, cur_edges in edges_loaded[src_hook_point].items():
                if dst_hook_point in nodes_indices_loaded.keys():
                    for dst_ind in nodes_indices_loaded[dst_hook_point]:
                        if src_hook_point in nodes_indices_loaded.keys():
                            all_values = cur_edges[dst_ind]
                            all_values_abs = all_values.abs()
                            values = cur_edges[dst_ind][initial_indices].tolist()
                            
                            look_at_indices = initial_indices
                            look_at_values,look_at_indices = zip(*sorted(zip(values, look_at_indices)))
                            compare_values_abs, compare_indices = torch.topk(all_values_abs, len(values))
                            compare_values = all_values[compare_indices]
                            compare_values, compare_indices = zip(*sorted(zip(compare_values.tolist(), compare_indices.tolist())))
                            # print(src_hook_point, dst_hook_point, dst_ind)
                            # print("FOUND keeping these:", look_at_values)
                            # print(look_at_indices)
                            # print("But there is also:", compare_values)
                            # print(compare_indices)

                            max_feature = saes[src_hook_point].d_sae 
                            for new_i, new_v in zip(compare_indices,compare_values):
                                if np.abs(new_v)>1e-4:
                                    if new_i not in nodes_indices_loaded[src_hook_point] and new_i!=max_feature:
                                        nodes_indices_loaded[src_hook_point].append(new_i)
                                        nodes_values_loaded[src_hook_point].append(-999) #TODO compute
    #print(nodes_indices_loaded)             
    #exit(0)


    ###### GENERATE IMAGES #############################################

    os.makedirs(output_folder, exist_ok=True)


    if do_generate_images:
        
        # The first step is to get highly activating images for each node as a crude way of autointerpreting
        ### Get indices of the images for each feature

        # remove the error term it needs to be handled separately 
        pruned_feature_ids = {}
        pruned_feature_vals = {}
        error_features = {}
        error_vals = {}
        for hook_point in nodes_indices_loaded.keys():
            max_feature = saes[hook_point].d_sae 
            pruned_feature_ids[hook_point] = []
            pruned_feature_vals[hook_point] = []
            error_features[hook_point] = []
            error_vals[hook_point] = []
             # the max_feature represents sae error, so it has no associated autointerp, so we represent it with black image 
            for i, v in zip(nodes_indices_loaded[hook_point], nodes_values_loaded[hook_point]):
                if i != max_feature:
                    pruned_feature_ids[hook_point].append(i)
                    pruned_feature_vals[hook_point].append(v)
                else:
                    error_features[hook_point].append(i)
                    error_vals[hook_point].append(v)
        top_activations_per_feature = find_top_activations(
                    dataloader, model, saes,
                    pruned_feature_ids, {k: [False]*len(ids) for k,ids in pruned_feature_ids.items()}, batch_size=batch_size, top_k=16, max_samples=50_000,
                )
        # now save the images in a grid format
        for hook_point in nodes_indices_loaded.keys():
            sae = saes[hook_point]

            
            

            root_name = re.sub(r'\b\d+\b', lambda x: f"{int(x.group()):02d}", hook_point)
            visualize_top_activating_images(output_folder, root_name, model, sae, top_activations_per_feature[hook_point], pruned_feature_ids[hook_point], dataset, visualize_dataset, device)

            if error_features[hook_point]:
                for e in error_features[hook_point]:
                    # make a dummy image!

                    # Create a black image of size 100x100
                    black_image = Image.new("RGB", (100, 100), (0, 0, 0))

                    black_image_path =  os.path.join(output_folder, f"{root_name}_{e}.{output_image_format}")

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
            image_path = os.path.join(output_folder, f"{root_name}.{output_image_format}")


            assert os.path.exists(image_path), f"could not find {image_path} do_generate_images: {do_generate_images}"
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
    output_file = os.path.join(output_folder, "circuit.html")  # must be saved in same folder as images
    net.write_html(output_file)



    if auto_open_browse_on_completion:
        webbrowser.open(output_file) 

        ###### END CREATE GRAPH #################################################

