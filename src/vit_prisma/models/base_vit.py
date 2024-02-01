import logging

import torch
import torch.nn as nn

from transformers import ViTForImageClassification

from vit_prisma.models.layers.patch_embedding import PatchEmbedding
from vit_prisma.models.layers.position_embedding import PosEmbedding
from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.models.layers.attention import Attention
from vit_prisma.models.layers.transformer_block import TransformerBlock

from vit_prisma.training.training_dictionary import activation_dict, initialization_dict
# from vit_prisma.models.prisma_net import PrismaNet
from vit_prisma.prisma.hook_point import HookPoint
from vit_prisma.prisma.hooked_root_module import HookedRootModule

from vit_prisma.configs import HookedViTConfig

from vit_prisma.prisma.activation_cache import ActivationCache

from vit_prisma.utils.prisma_utils import convert_pretrained_model_config

from typing import Union, Dict, List, Tuple, Optional, Literal

from jaxtyping import Float, Int

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

class HookedViT(HookedRootModule):
    """
    Base vision model.
    Based on 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
    Adapted from TransformerLens: https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/HookedTransformer.py
    Inspiration also taken from the timm library.
    """

    def __init__(
            self,
            cfg: HookedViTConfig,
            move_to_device: bool = True,
    ):
        """
        Model initialization
        """

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedViT object. If you want to load a "
                "pretrained model, use HookedViT.from_pretrained() instead."
            )
        self.cfg = cfg

        # ClS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model))

        # Patch embeddings
        self.embed = PatchEmbedding(self.cfg)
        self.hook_embed = HookPoint()

        # Position embeddings
        self.pos_embed = PosEmbedding(self.cfg)
        self.hook_pos_embed = HookPoint()

        # Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.n_layers)
            ]
        )
        # Final layer norm
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln_final = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization type: {self.cfg.normalization_type}")

        # Final classification head
        self.head = nn.Linear(self.cfg.d_model, self.cfg.n_classes)

        # Initialize weights
        self.init_weights()

        # Set up HookPoints
        self.setup()

    def forward(self,
            input: Union[
            str,
            List[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos d_model"],
        ],):

        batch_size = input.shape[0]

        # Embedding
        embed = self.hook_embed(self.embed(input))


        if self.cfg.classification_type == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # CLS token for each item in the batch
            embed = torch.cat((cls_tokens, embed), dim=1) # Add to embedding

        pos_embed = self.hook_pos_embed(self.pos_embed(input))
        full_embeddings = embed + pos_embed

        # Blocks
        for block in self.blocks:
            residual = block(full_embeddings)

        # Final layer norm
        x = self.ln_final(residual)

        if self.cfg.classification_type == 'gaap':  # GAAP
            x = x.mean(dim=1)
        elif self.cfg.classification_type == 'cls':  # CLS token
            x = x[:, 0]
            
        return x if self.cfg.return_type == 'pre_logits' else self.head(x)

    def init_weights(self):
        if self.cfg.classification_type == 'cls':
            nn.init.normal_(self.cls_token, std=self.cfg.cls_std)
        # nn.init.trunc_normal_(self.position_embedding, std=self.cfg.pos_std)   
        if self.cfg.weight_type == 'he':
            for m in self.modules(): 
                if isinstance(m, PosEmbedding):
                    nn.init.normal_(m.W_pos, std=self.cfg.pos_std)
                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch n_classes"],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict
        
    def tokens_to_residual_directions(self):
        '''
        Logit-lens related funtions not implemented; see how we can implement a vision equivalent.
        '''
        
        pass 

    @classmethod
    def from_pretrained(
        cls, 
        model_name: str,
        fold_ln: Optional[bool] = True,
        center_writing_weights: Optional[bool] = True,
        center_unembed: Optional[bool] = True,
        refactor_factored_attn_matrices: Optional[bool] = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[ViTForImageClassification] = None,
        device: Optional[Union[str, torch.device]] = None,
        n_devices: Optional[int] = 1,
        move_to_device: Optional[bool] = True,
        fold_value_biases: Optional[bool] = True,
        default_prepend_bos: Optional[bool] = True,
        default_padding_side: Optional[Literal["left", "right"]] = "right",
        dtype="float32",
        **from_pretrained_kwargs,
    ) -> "HookedViT":
        
        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if isinstance(dtype, str):
            # Convert from string to a torch dtype
            dtype = DTYPE_FROM_STRING[dtype]
        if "torch_dtype" in from_pretrained_kwargs:
            # For backwards compatibility with the previous way to do low precision loading
            # This should maybe check the user did not explicitly set dtype *and* torch_dtype
            dtype = from_pretrained_kwargs["torch_dtype"]

        if (
            (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
            or dtype == torch.float16
        ) and device in ["cpu", None]:
            logging.warning(
                "float16 models may not work on CPU. Consider using a GPU or bfloat16."
            )
        
        cfg = convert_pretrained_model_config(
            model_name,
            is_timm=True,
        )

        model = cls(cfg, move_to_device=False)

        logging.warning(
            'The model\'s computation graph is constructed, but the weights are not loaded. '
        )

        return model

