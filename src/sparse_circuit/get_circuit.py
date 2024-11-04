
import torch 

from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
from typing import List, Dict, Tuple
import einops
from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
from vit_prisma.dataloaders.imagenet_dataset import ImageNetValidationDataset
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.sae import SparseAutoencoder
from typing import List 
import urllib.request
from fancy_einsum import einsum
import torchvision
from huggingface_hub import hf_hub_download
from vit_prisma.utils.load_model import load_model
import open_clip
from sparse_circuit.sparse_act import SparseAct

#TODO if sae was refactored might be easier to do
def sae_decode(sae, acts, original_input):
    # need to get normalization parameters
    sae.run_time_activation_norm_fn_in(original_input)

    # now run the decoding steps

    sae_out = einops.einsum(
        acts,
        sae.W_dec,
        "... d_sae, d_sae d_in -> ... d_in",
    )
    + sae.b_dec


    sae_out = sae.run_time_activation_norm_fn_out(sae_out)

    return sae_out


# Note this is the no pair version! (no 'patch') (i.e. patch is 0's)
def get_circuit_nodes(clean_inputs, patch_inputs, model, saes, metric_fn, aggregation='sum', ig_steps=10):

    assert patch_inputs == None, "it's not too much extra work but so far haven't done the patch inputs case"
    
    # start by computing patching effect using ig 

    # run the model and cache all sparse activations 
    hidden_states_clean = {}
    with torch.no_grad():
        for hook_point in saes.keys():
            names_filter = list(saes.keys())
            _, cache = model.run_with_cache(clean_inputs, names_filter=names_filter)

            
            for hook_point, sae in saes.items():

                # run the hook while caching the feature acts
                sae_inp = cache[hook_point]
                
                sae_out, feature_acts, *_ = sae(sae_inp)

                error = sae_inp - sae_out

                hidden_states_clean[hook_point] = SparseAct(act =feature_acts, res=error)


            #TODO we don't have patch_inputs so currently don't need clean metric
                


    if patch_inputs is None:
        # just zero
        hidden_states_patch = {
        k : SparseAct(act=torch.zeros_like(v.act), res=torch.zeros_like(v.res)) for k, v in hidden_states_clean.items()
    }
    else:
        raise NotImplementedError
        #TODO normally would get the patched activations but see the assert above, currently not implemented
    

    #get the effect (i.e. delta*grad) of ablating each feature
    effects = {}
   # deltas = {}
   # grads = {}

    for hook_point, sae in saes.items():
        clean_state = hidden_states_clean[hook_point]

        patch_state = hidden_states_patch[hook_point]
        
        metrics = []
        fs = []

        for step in range(1, ig_steps+1): #TODO this is inline with the paper but not the github code (but probably doesn't matter either way, just mentioning it, code is range(ig_steps))
            alpha = step / ig_steps
            f = (1 - alpha) * clean_state + alpha * patch_state 
            f.act.requires_grad = True
            f.res.requires_grad = True 
            f.act.retain_grad()
            f.res.retain_grad()
            
            fs.append(f)

            def replacement_hook(x, hook):
                return sae_decode(sae, f.act, x) + f.res 
            
            model_output = model.run_with_hooks(clean_inputs, fwd_hooks =[(hook_point, replacement_hook)])
            metric = metric_fn(model_output)

            metrics.append(metric)
        metric = sum([m for m in metrics])
        metric.sum().backward(retain_graph=True)


        mean_grad = sum([f.act.grad for f in fs]) / ig_steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / ig_steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)


        delta = (patch_state - clean_state).detach()
 

        effect = grad @ delta # residual gets contracted

        effects[hook_point] = effect
       # deltas[hook_point] = delta
        #grads[hook_point] = grad


    # TODO all the edges stuff! 
        
    # sum over patchs and take average over batch
    #TODO this can be sped up by keeping it in tensor form.
    nodes = effects
    if aggregation == 'sum':
        for k in nodes:
            nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}


    #nodes['y'] = None # TODO total effect (only used for pair case)
        
    return nodes 




        
    
    

                





