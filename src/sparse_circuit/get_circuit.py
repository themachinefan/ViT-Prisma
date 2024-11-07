
import torch 
from collections import defaultdict
from typing import List, Dict, Tuple, Union
import einops
from tqdm import tqdm 
from sparse_circuit.sparse_act import SparseAct
from vit_prisma.sae.sae import SparseAutoencoder
## adapted from https://github.com/saprmarks/feature-circuits 
def jvp(
        inputs,
        model,
        downstream_sae:SparseAutoencoder,
        downstream_features,
        upstream_sae:SparseAutoencoder,
        left_vec : Union[SparseAct, Dict[int, SparseAct]],
        right_vec : SparseAct,
):
    """
    Return a sparse shape [# downstream features + 1, # upstream features + 1] tensor of Jacobian-vector products.
    """
    
    b, s, n_feats_p1 = downstream_features.shape

    if torch.all(downstream_features == 0):
        return torch.sparse_coo_tensor(
            torch.zeros((2 * downstream_features.dim(), 0), dtype=torch.long), 
            torch.zeros(0), 
            size=(b, s,n_feats_p1, b, s, n_feats_p1)
        ).to(right_vec.act.device)


    print("COMPUTING FOR ", len(downstream_features), "features!")
    vjv_values = {}

    downstream_hook = downstream_sae.cfg.hook_point
    upstream_hook = upstream_sae.cfg.hook_point


    # x = upstream_submod.get_activation()
    # x_hat, f = upstream_dict.hacked_forward_for_sfc(x) # hacking around an nnsight bug
    # x_res = x - x_hat
    # upstream_submod.set_activation(x_hat + x_res)
    # upstream_act = SparseAct(act=f, res=x_res).save()
    
    # y = downstream_submod.get_activation()
    # y_hat, g = downstream_dict.hacked_forward_for_sfc(y) # hacking around an nnsight bug
    # y_res = y - y_hat
    # downstream_act = SparseAct(act=g, res=y_res)

    # to_backprops = (left_vec @ downstream_act).to_tensor()


    upstream_act = None 
    downstream_act = None 
    upstream_error = None 
    def upstream_hook_fn(x, hook):
        x = x.detach()
        nonlocal upstream_act, upstream_error

        upstream_sae_out, upstream_feature_acts, *_ = upstream_sae(x)



        upstream_error = (x - upstream_sae_out).detach()
        upstream_error.requires_grad = True 
        upstream_act = SparseAct(act =upstream_feature_acts, res=upstream_error)
        return upstream_sae_out + upstream_error
    
    def downstream_hook_fn(x, hook):
        nonlocal downstream_act
        downstream_sae_out, downstream_feature_acts, *_ = downstream_sae(x)


        downstream_act = SparseAct(act =downstream_feature_acts, res=x - downstream_sae_out)
        return x
            
    _ = model.run_with_hooks(inputs, fwd_hooks =[(upstream_hook, upstream_hook_fn), 
                                                 (downstream_hook, downstream_hook_fn)])

    upstream_act.act.retain_grad()
    upstream_act.res.retain_grad()

    to_backprops = (left_vec @ downstream_act).to_tensor()

    for downstream_feat_idx in downstream_features.nonzero():
        # in full version need intermediate stop grads here but should be ok since only doing residual stream

        to_backprops[tuple(downstream_feat_idx)].backward(retain_graph=True)


        #TODO the code is meant to get gradient info from res even though it's detached but am I doing it right?
        vjv = (upstream_act.grad @ right_vec).to_tensor()

        vjv_values[downstream_feat_idx] = vjv
    

    vjv_indices = torch.stack(list(vjv_values.keys()), dim=0).T
    vjv_values = torch.stack([v for v in vjv_values.values()], dim=0)

    return torch.sparse_coo_tensor(vjv_indices, vjv_values, size=(b, s, n_feats_p1, b, s, n_feats_p1))



# def jvpold(
#     input,
#     model,
#     dictionaries,
#     downstream_submod,
#     downstream_features,
#     upstream_submod,
#     left_vec: SparseAct,
#     right_vec: SparseAct,
# ):

#     downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]
#     b, s, n_feats = downstream_features.act.shape

#     # if t.all(downstream_features.to_tensor() == 0):
#     #     return t.sparse_coo_tensor(
#     #         t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long), 
#     #         t.zeros(0), 
#     #         size=(b, s, n_feats+1, b, s, n_feats+1)
#     #     ).to(model.device)


#     vjv_values = {}

#     x = upstream_submod.get_activation()
#     x_hat, f = upstream_dict.hacked_forward_for_sfc(x) # hacking around an nnsight bug
#     x_res = x - x_hat
#     upstream_submod.set_activation(x_hat + x_res)
#     upstream_act = SparseAct(act=f, res=x_res).save()
    
#     y = downstream_submod.get_activation()
#     y_hat, g = downstream_dict.hacked_forward_for_sfc(y) # hacking around an nnsight bug
#     y_res = y - y_hat
#     downstream_act = SparseAct(act=g, res=y_res)

#     to_backprops = (left_vec @ downstream_act).to_tensor()

#     for downstream_feat_idx in downstream_features.to_tensor().nonzero():
#         # stop grad
#         # for submodule in intermediate_stopgrads:
#         #     submodule.stop_grad()
#         x_res.grad = torch.zeros_like(x_res.grad)

#         vjv = (upstream_act.grad @ right_vec).to_tensor()
#         to_backprops[tuple(downstream_feat_idx)].backward(retain_graph=True)

#         vjv_values[downstream_feat_idx] = vjv.save() # type: ignore


#     vjv_indices = torch.stack(list(vjv_values.keys()), dim=0).T
#     vjv_values = torch.stack([v.value for v in vjv_values.values()], dim=0)

#     return torch.sparse_coo_tensor(vjv_indices, vjv_values, size=(b, s, n_feats+1, b, s, n_feats+1))

###### utilities for dealing with sparse COO tensors ######
def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = torch.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return torch.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return torch.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return torch.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)

######## end sparse tensor utilities ########






  
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
def get_circuit(clean_inputs, patch_inputs, model, saes, metric_fn, aggregation='sum', ig_steps=10, nodes_only=False,
    node_abs_threshold = 0.02,
    node_max_per_hook=None,
    
    ):

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
    deltas = {}
    grads = {}

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
        deltas[hook_point] = delta
        grads[hook_point] = grad


        




        
    # sum over patchs and take average over batch
    #TODO this can be sped up by keeping it in tensor form.
    if nodes_only:
        nodes = effects
        if aggregation == 'sum':
            for k in nodes:
                nodes[k] = nodes[k].sum(dim=1)
            nodes = {k : v.mean(dim=0) for k, v in nodes.items()}


        #nodes['y'] = None # TODO total effect (only used for pair case)
            
        return nodes, None 
    

    # figure out which nodes are going to be kept. Using an abs threshold and a std threshold

    features = {} 

    for hook_point, effect in effects.items():

        tensor_effect = effect.to_tensor()

        abs_effects = tensor_effect.abs()
        
        # Initialize the mask as a tensor of True values
        mask = torch.ones_like(abs_effects, dtype=torch.bool)
        
        # Apply the top-k threshold if top_k is not None
        if node_max_per_hook is not None and node_max_per_hook <= abs_effects.numel():
            top_k_values, _ = torch.topk(abs_effects.flatten(), node_max_per_hook)
            top_k_threshold = top_k_values[-1]  # Smallest value in the top-k
            mask &= abs_effects >= top_k_threshold  # Update mask with top-k condition
        
        # Apply the node threshold if node_threshold is not None
        if node_abs_threshold is not None:
            mask &= abs_effects > node_abs_threshold  # Update mask with node_threshold condition
        
        # Store the final mask in the dictionary
        features[hook_point] = mask

        print("new featres", mask.shape, len(mask.nonzero()))


      #  features[hook_point] =
 #   features_by_submod = {
 #       submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
  #  }

    

    # get the edges 
    #TODO assuming all resid post for now! need to extend to mlp and attn as in paper 
    edges = defaultdict(lambda:{})

    for layer in reversed(range(len(saes))):
        if layer == 0: #needs a embed sae 
            break 
        resid_hook_point = f"blocks.{layer}.hook_resid_post"
        prev_resid_hook_point = f"blocks.{layer-1}.hook_resid_post"


        RR_edge = jvp(
            clean_inputs,
            model,
            saes[resid_hook_point],
            features[resid_hook_point],
            saes[prev_resid_hook_point],
            grads[resid_hook_point],
            deltas[prev_resid_hook_point],
            
        )

        edges[prev_resid_hook_point][resid_hook_point] = RR_edge

    nodes = effects
#     for child in edges:
#         # get shape for child
#         bc, sc, fc = nodes[child].act.shape
#         for parent in edges[child]:
#             weight_matrix = edges[child][parent]
#             if parent == 'y':
#                 weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
#             else:
#                 bp, sp, fp = nodes[parent].act.shape
#                 assert bp == bc
#                 weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
#             edges[child][parent] = weight_matrix


#     if aggregation == 'sum':
# #         # aggregate across sequence position
#         for child in edges:
#             for parent in edges[child]:
#                 weight_matrix = edges[child][parent]
#                 if parent == 'y':
#                     weight_matrix = weight_matrix.sum(dim=1)
#                 else:
#                     weight_matrix = weight_matrix.sum(dim=(1, 4))
#                 edges[child][parent] = weight_matrix
#         for node in nodes:
#             if node != 'y':
#                 nodes[node] = nodes[node].sum(dim=1)

#         # aggregate across batch dimension
#         for child in edges:
#             bc, fc = nodes[child].act.shape
#             for parent in edges[child]:
#                 weight_matrix = edges[child][parent]
#                 if parent == 'y':
#                     weight_matrix = weight_matrix.sum(dim=0) / bc
#                 else:
#                     bp, fp = nodes[parent].act.shape
#                     assert bp == bc
#                     weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
#                 edges[child][parent] = weight_matrix
#         for node in nodes:
#             if node != 'y':
#                 nodes[node] = nodes[node].mean(dim=0)
    
#     elif aggregation == 'none':

#         # aggregate across batch dimensions
#         for child in edges:
#             # get shape for child
#             bc, sc, fc = nodes[child].act.shape
#             for parent in edges[child]:
#                 weight_matrix = edges[child][parent]
#                 if parent == 'y':
#                     weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
#                     weight_matrix = weight_matrix.sum(dim=0) / bc
#                 else:
#                     bp, sp, fp = nodes[parent].act.shape
#                     assert bp == bc
#                     weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
#                     weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
#                 edges[child][parent] = weight_matrix
#         for node in nodes:
#             nodes[node] = nodes[node].mean(dim=0)

#     else:
#         raise ValueError(f"Unknown aggregation: {aggregation}")
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                continue
                # bp, sp, fp = nodes[parent].act.shape
                # assert bp == bc
                # try:
                #     weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                # except:
                #     breakpoint()
            edges[child][parent] = weight_matrix
    
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    # weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return nodes, edges, features
