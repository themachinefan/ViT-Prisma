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
    
#TODO clean up and expose arguments.
def setup(device, debug=False):
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
    
    }
    if debug:
        for i in range(2,12):
            del resid_post_repo_ids[i]


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
    imagenet_dataset_path = r"F:/prisma_data/imagenet-object-localization-challenge"

    output_folder = r"F:\ViT-Prisma_fork\data\circuit_output"
    output_name = "testing_stuff_2"
    num_workers = 3
    batch_size = 16
    device = "cuda"
    ig_steps = 10
    num_examples = 250
    nodes_only = False 
    only_these_labels=[281, 282, 283, 284, 285]
    top_k = 25 #  saves all nodes but only looks for edges between the top_k of each (can be set to None)
    node_threshold = None #  saves all nodes but only looks for edges between those above this threshold (can be set to None)


    debug = False
    if debug:
        num_workers = 0 
        batch_size = 2 
        num_examples = 10
        only_these_labels=[281, 282, 283, 284, 285]
        top_k = 3



    node_name = f"{output_name}_nodes.pt"
    edges_name = f"{output_name}_edges.pt"
    features_name = f"{output_name}_features.pt"
    dataset = get_imagenet_val_dataset(imagenet_dataset_path, only_these_labels=only_these_labels)
   # visualize_dataset = get_imagenet_val_dataset_visualize(imagenet_dataset_path, only_these_labels=only_these_labels)
    


    model, saes, model_name = setup(device, debug=debug)
    
    clip_text_stuff = ClipTextStuff(model_name, device)





   
    #TODO use data for a specific concept in mind. 

    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    num_examples = min(num_examples, batch_size*(len(dataset)//batch_size))


    cat = clip_text_stuff.get_text_embeds(["cat"])[0] # d_embed
    #TODO should I use the logit scale for this? clip_text_stuff.logit_scale.exp()


    batch_i = 0
    running_nodes = None 
    running_edges = None
    import time 
    tic = time.perf_counter()
    for batch in tqdm(dataloader):
        if batch_i*batch_size >= num_examples:
            break 
        batch_i += 1
        #defined here since in some cases the metric changes each batch (if the answer changes)
        def metric_fn(model_output):
            scores = einsum( "I D, D -> I", model_output, cat)
            return scores 
        
        images = batch[0]

        images = images.to(device)
        
        #return  get_circuit(images, None, model, saes, metric_fn, ig_steps=ig_steps, nodes_only=nodes_only, node_abs_threshold=node_threshold, node_max_per_hook=top_k)

        nodes, edges = get_circuit(images, None, model, saes, metric_fn, ig_steps=ig_steps, nodes_only=nodes_only, node_abs_threshold=node_threshold, node_max_per_hook=top_k)


        #TODO this can be sped up by keeping it in tensor form.

        # add up all the nodes! (also undo the mean from get_circuit_nodes, turning it in to a sum, could just not do it in the first place but this is consistent with the code and might make more sense when the edge part is done)
        if running_nodes is None:
            running_nodes = {k : batch_size * nodes[k] for k in nodes.keys() if k != 'y'}
            if edges is not None:
                running_edges = { k : { kk : batch_size * edges[k][kk] for kk in edges[k].keys() } for k in edges.keys()}

        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += batch_size * nodes[k]
            if edges is not None:
                for k in edges.keys():
                    for v in edges[k].keys():
                        running_edges[k][v] += batch_size * edges[k][v]

        del nodes 
        del edges 
    # take a global average! 
    nodes = {k : v / num_examples for k, v in running_nodes.items()}
    if running_edges is not None:
        edges = {k : {kk : 1/num_examples * v for kk, v in running_edges[k].items()} for k in running_edges.keys()}
    final_nodes = {}


    #NOTE I'm saving ALL nodes because how to threshold is not totally clear and I want to experiment 
    for k,v in nodes.items():

    
        print(v.act.shape, v.resc.shape)

        final_nodes[k] =  torch.cat((v.act, v.resc), dim=0)

    final_edges = {}

    for k1,v1 in edges.items():
        if k1 not in final_edges.keys():
            final_edges[k1] = {}
        for k2, v2 in v1.items():
            final_edges[k1][k2] = v2


    torch.save(final_nodes, os.path.join(output_folder, node_name))
    torch.save(final_edges, os.path.join(output_folder, edges_name))
   # torch.save(features, os.path.join(output_folder,features_name ))
    print("WOW that took", time.perf_counter() - tic, "seconds")
    #final_nodes_loaded = torch.load(save_path)

if __name__ == "__main__":
    main()
