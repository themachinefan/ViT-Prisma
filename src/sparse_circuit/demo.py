import torch 

from torch.utils.data import DataLoader
import os 
from tqdm import tqdm
from typing import List, Dict, Tuple
from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import ImageNetValidationDataset
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
from vit_prisma.sae.sae import SparseAutoencoder
from typing import List 
from fancy_einsum import einsum
import torchvision
from huggingface_hub import hf_hub_download
from vit_prisma.utils.load_model import load_model
import open_clip
from sparse_circuit.get_circuit import get_circuit
import time 
from vit_prisma.sae.config import VisionModelSAERunnerConfig


#NOTE THIS IS THE NO PAIR VERSION. 


##### HELPER FUNCTIONS 
def get_imagenet_val_dataset(dataset_path, only_these_labels=None):

    data_transforms = get_clip_val_transforms()
    imagenet_paths = setup_imagenet_paths(dataset_path)
    return ImageNetValidationDataset(imagenet_paths['val'], 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'], 
                                    data_transforms, return_index=True, only_these_labels=only_these_labels
    )

def get_imagenet_val_dataset_visualize(dataset_path, only_these_labels=None):
    imagenet_paths = setup_imagenet_paths(dataset_path)

    return ImageNetValidationDataset(imagenet_paths['val'], 
                                imagenet_paths['label_strings'], 
                                imagenet_paths['val_labels'],
                                torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),]), return_index=True, only_these_labels=only_these_labels)

def load_sae_config(repo_id,config_name ="config.json"):
    config_path = hf_hub_download(repo_id, config_name)

    from vit_prisma.sae.config import VisionModelSAERunnerConfig
    return VisionModelSAERunnerConfig.load_config(config_path)  
     
    pass 

def load_sae(repo_id="Prisma-Multimodal/8e32860c-clip-b-sae-gated-all-tokens-x64-layer-9-mlp-out-v1", file_name="n_images_2600058.pt", config_name ="config.json"):
    """
    Load and test SAE from HuggingFace
    """
    print(f"Loading model from {repo_id}...")


    sae_path = hf_hub_download(repo_id, file_name)
    hf_hub_download(repo_id, config_name)

    sae = SparseAutoencoder.load_from_pretrained(sae_path) # This now automatically gets config.json and converts into the VisionSAERunnerConfig object



    # Move to available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
        
    return sae



class ClipTextStuff:

    def __init__(self, model_name, device="cuda"):

        self.device= device 

        hf_model_name = model_name.replace("open-clip:", "hf-hub:")
        full_clip_model, _ = open_clip.create_model_from_pretrained(hf_model_name)
        self.clip_processor = open_clip.get_tokenizer(hf_model_name)

        self.full_clip = full_clip_model
        self.logit_scale = full_clip_model.logit_scale



    def get_text_embeds(self, list_of_strings:List[str]):




        text_inputs = self.clip_processor(list_of_strings)
        text_embeddings = self.full_clip.encode_text(text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings.to(self.device)
        return text_embeddings

## fake sae that really just returns the neurons of whichever layer it's set up for. 
## mimics the format of our sparseautoencoder so it can just be plugged into the circuit code
class FakeSae(torch.nn.Module):

    def __init__(self,cfg: VisionModelSAERunnerConfig):
        super().__init__()
        self.cfg = cfg
        self.cfg.d_sae = self.cfg.d_in # since this is just neurons 
        self.d_sae = self.cfg.d_in


        # used to fake decoding and encoding
        self.W_dec = torch.nn.Parameter(
                torch.eye(self.cfg.d_in, dtype=self.cfg.dtype, device=self.cfg.device)
                )
        self.b_dec = torch.nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype, device=self.cfg.device)
        )

        self.W_enc = torch.nn.Parameter(
                torch.eye(self.cfg.d_in, dtype=self.cfg.dtype, device=self.cfg.device)
                )
        self.b_enc = torch.nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype, device=self.cfg.device)
        )



    
    def forward(self, x):
        # this just returns x and 'features' which is also.. just x.plus a bunch of Nones to mimic the sae 
        fake_sae_activations = x + 0

        fake_sae_out = fake_sae_activations + 0
        return fake_sae_out, fake_sae_activations, None, None, None, None, None 
    
    def run_time_activation_norm_fn_in(self, x):
        return x 
    def run_time_activation_norm_fn_out(self,x):
        return x 

def setup_saes_and_model(device,
              # I choose the ones that seemed like a good balance between xvar and l1 
              # that ended up being the ones with xvar ~= 80% 
              # I don't have an ironclad reason for making this choice though. 
              resid_post_repo_ids = {
        0: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-0-hook_resid_post-l1-0.0001",
        1: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-1-hook_resid_post-l1-0.0001",
        2: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-2-hook_resid_post-l1-0.0001",
        3: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-3-hook_resid_post-l1-0.0001",
        4: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-4-hook_resid_post-l1-0.0001",
        5: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-5-hook_resid_post-l1-0.0001",
        6: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-6-hook_resid_post-l1-8e-05",
        7: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_resid_post-l1-0.0001",
        8: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_resid_post-l1-0.0001",
        9: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_resid_post-l1-8e-05",
        10: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_resid_post-l1-8e-05",
        11: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-8e-05",
    },
           debug=False,
           only_these_layers=None,
           use_neurons=False # if true will load 'dummy' saes that actually just use neurons
           ):

    if debug:
        for i in range(2,12):
            del resid_post_repo_ids[i]
    elif only_these_layers is not None:
        for i in range(0,12):
            if i not in only_these_layers:
                del resid_post_repo_ids[i]


    if use_neurons:
        # load the cfg to create dummy saes but dont load the sae as a whole
        saes:Dict[str, FakeSae] = {} 
        model_name = None 
        model_class_name = None  
        for layer,repo_id in resid_post_repo_ids.items():
            sae_cfg = load_sae_config(repo_id)
            fake_sae = FakeSae(sae_cfg)

            fake_sae.cfg.device = device
            fake_sae = fake_sae.to(device)
            assert fake_sae.cfg.layer_subtype == 'hook_resid_post', 'Currently only set up to hande hook_resid_post, in particular need to update the edge computation. Not too hard but requires both and mlp and attn saes'

            assert fake_sae.cfg.architecture == "standard", "Haven't checked what changes are needed for gated if any"

            saes[f"blocks.{layer}.hook_resid_post"] = fake_sae


            if model_name is None:
                model_name = fake_sae.cfg.model_name 
                model_class_name = fake_sae.cfg.model_class_name
                
            assert model_name == fake_sae.cfg.model_name, "One of the saes is for a different model!"
    else:
        # I'm only looking at resid_post just to test things out. The original paper using resid_post, mlp_out and the output of the attention blocks 
        saes:Dict[str, SparseAutoencoder] = {} # hookpoint to sae in some kinda sensible order 
        model_name = None 
        model_class_name = None 
        for layer,repo_id in resid_post_repo_ids.items():
            sae = load_sae(repo_id)
            
            sae.cfg.device = device
            sae = sae.to(device)
            assert sae.cfg.layer_subtype == 'hook_resid_post', 'Currently only set up to hande hook_resid_post, in particular need to update the edge computation. Not too hard but requires both and mlp and attn saes'

            assert sae.cfg.architecture == "standard", "Haven't checked what changes are needed for gated if any"

            saes[f"blocks.{layer}.hook_resid_post"] = sae


            if model_name is None:
                model_name = sae.cfg.model_name 
                model_class_name = sae.cfg.model_class_name
                
            assert model_name == sae.cfg.model_name, "One of the saes is for a different model!"

    # get the model 
    model = load_model(model_class_name, model_name)
    model = model.to(device)

    return  model, saes, model_name


def main():

    ### SETUP #########################################

    # args
    imagenet_dataset_path = r"F:/prisma_data/imagenet-object-localization-challenge"

    output_folder = r"F:\ViT-Prisma_fork\data\circuit_output"
    output_name = "testing_stuff_3_saes"
    num_workers = 3
    node_batch_size = 16 
    edge_batch_size = 4 #NOTE smaller is actually better aside from speed
    device = "cuda"
    ig_steps = 10
    num_examples = 250
    nodes_only = False 
    only_these_labels=[281, 282, 283, 284, 285] # only use these classes from imagenet1k
    top_k = 10 # save at most top_k nodes per layer
    node_threshold_std = 3 # keep nodes x std deviations from mean 
    only_these_layers = [6, 7,8,9,10,11] #earlier layers don't seems as easy to interpret so for simplicity using layer layers only
    tokens_per_node = 2 #NOTE make this as big as you can. how many tokens to use per node during edge computation 
    use_neurons = False # Use neurons instead of sae activations!


    debug = False
    if debug:
        num_workers = 0 
        node_batch_size = 2 
        edge_batch_size = 2 
        num_examples = 5
        top_k = 3



    # get the model and the saes
    model, saes, model_name = setup_saes_and_model(device, debug=debug, only_these_layers=only_these_layers, use_neurons=use_neurons)
    
    # get the text embeddings for tinyclip
    clip_text_stuff = ClipTextStuff(model_name, device)
   
    # get the dataloader
    dataset = get_imagenet_val_dataset(imagenet_dataset_path, only_these_labels=only_these_labels)
    
    

    ##### SETUP END #########################################################
 
    node_name = f"{output_name}_nodes.pt"
    parsed_node_name = f"{output_name}_nodes_parsed.pt"
    edges_name = f"{output_name}_edges.pt"
    

    cat = clip_text_stuff.get_text_embeds(["cat"])[0] # d_embed
    #TODO should I use the logit scale for this? clip_text_stuff.logit_scale.exp()




    # NOTE It's slower but seems best to first get values for all nodes, then choose which ones to keep and finally find edges for those nodes

    ### FIRST GET THE NODES 
    print("FINDING NODES")
    batch_i = 0
    running_nodes = None 
    dataloader =  DataLoader(dataset, batch_size=node_batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    num_examples = min(num_examples, node_batch_size*(len(dataset)//node_batch_size))
    
    tic = time.perf_counter()


    for batch in tqdm(dataloader):
        if batch_i*node_batch_size >= num_examples:
            break 
        batch_i += 1

        # metric used!
        #defined here since in some cases the metric changes each batch (if the answer changes)
        def metric_fn(model_output):
            scores = einsum( "I D, D -> I", model_output, cat)
            return scores 
        
        images = batch[0]

        images = images.to(device)
     
        nodes, _ = get_circuit(images, None, model, saes, metric_fn, ig_steps=ig_steps, nodes_only=True)


    
        # add up all the nodes! (also undo the mean from get_circuit_nodes, turning it in to a sum, could just not do it in the first place but this is consistent with the code and might make more sense when the edge part is done)
        if running_nodes is None:
            running_nodes = {k : node_batch_size * nodes[k] for k in nodes.keys() if k != 'y'}
        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += node_batch_size * nodes[k]

        del nodes 

    # take a global average! 
    nodes = {k : v / num_examples for k, v in running_nodes.items()}


    final_nodes = {}
    for k,v in nodes.items():
        final_nodes[k] =  torch.cat((v.act, v.resc), dim=0)

    torch.save(final_nodes, os.path.join(output_folder, node_name))

    # parse the 'best' nodes to use to compute edges
    custom_node_indices = {}
    custom_node_values = {}
    for hook_point, results in final_nodes.items():

        results_abs = results.abs()
        mean = torch.mean(results)
        std = torch.std(results)


        # Find indices where A > mean + num_std * std
        if node_threshold_std is not None:
            indices = torch.nonzero(results > mean + node_threshold_std * std).squeeze()

            if indices.numel() == 0:
                # If indices is empty, take the index of the largest value in results
                indices = torch.argmax(results).unsqueeze(0)
            values_abs = results_abs[indices]            
            values = results[indices]
        else:
            values_abs = results_abs
            values = results
        if top_k is not None:
            if values_abs.dim() > 0 and len(values_abs) > top_k:
                og_amount = len(values_abs)   
                values_abs, top_k_indices = torch.topk(values_abs, top_k, largest=True)
                indices = indices[top_k_indices]
                values = values[top_k_indices]        

                print("removed", og_amount-top_k, 'nodes from the original', og_amount, 'in', hook_point)

        values_abs, sorted_indices = torch.sort(values_abs, descending=True)
        
        custom_node_indices[hook_point] = indices[sorted_indices].tolist()
        custom_node_values[hook_point] = values[sorted_indices].tolist()
    torch.save((custom_node_indices, custom_node_values), os.path.join(output_folder, parsed_node_name))

    del final_nodes
    del custom_node_values 

    if nodes_only:
        return 


    # Run again and find the edges for the nodes choosen above 
    print("FINDING EDGES")
    running_edges = None
    batch_i = 0

    dataloader =  DataLoader(dataset, batch_size=edge_batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    num_examples = min(num_examples, edge_batch_size*(len(dataset)//edge_batch_size))
    
    for batch in tqdm(dataloader):
        if batch_i*edge_batch_size >= num_examples:
            break 
        batch_i += 1

        # metric used!
        #defined here since in some cases the metric changes each batch (if the answer changes)
        def metric_fn(model_output):
            scores = einsum( "I D, D -> I", model_output, cat)
            return scores 
        
        images = batch[0]

        images = images.to(device)
     
        _, edges = get_circuit(images, None, model, saes, metric_fn, ig_steps=ig_steps, nodes_only=False,use_these_nodes=custom_node_indices, tokens_per_predetermined_node=tokens_per_node)

        # add up all the edges
        if running_edges is None:
            running_edges = { k : { kk : edge_batch_size * edges[k][kk] for kk in edges[k].keys() } for k in edges.keys()}

        else:
            for k in edges.keys():
                    for v in edges[k].keys():
                        running_edges[k][v] += edge_batch_size * edges[k][v]

        del edges 

        edges = {k : {kk : 1/num_examples * v for kk, v in running_edges[k].items()} for k in running_edges.keys()}


    final_edges = {}

    for k1,v1 in edges.items():
        if k1 not in final_edges.keys():
            final_edges[k1] = {}
        for k2, v2 in v1.items():
            final_edges[k1][k2] = v2


    torch.save(final_edges, os.path.join(output_folder, edges_name))

    print("WOW that took", time.perf_counter() - tic, "seconds")

if __name__ == "__main__":
    main()
