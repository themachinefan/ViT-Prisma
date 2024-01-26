import torch
import torch.nn as nn
from vit_prisma.models.layers.transformer_block import TransformerBlock

from vit_prisma.models.layers.patch_embedding import PatchEmbedding
from vit_prisma.models.layers.position_embeddding import PosEmbedding

from vit_prisma.training.training_dictionary import activation_dict, initialization_dict
from vit_prisma.models.prisma_net import PrismaNet
from vit_prisma.models.hook_points import HookedRootModule

class HookedViT(HookedRootModule):
    """
    Base vision model.
    Based on 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
    Adapted from TransformerLens: https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/HookedTransformer.py
    Inspiration also taken from the timm library.
    """

    def __init__(
            self,
            cfg: Union[HookedViTConfig, Dict],
            move_to_device: bool = True,
    ):
        """
        Model initialization
        """

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a "
                "pretrained model, use HookedTransformer.from_pretrained() instead."
            )
        self.cfg = cfg

        self.embed = PatchEmbedding(self.cfg)
        self.hook_embed = HookPoint()

        self.pos_embed = PosEmbedding(self.cfg)
        self.hook_pos_embed = HookPoint()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.transformer.num_layers)
            ]
        )
        # Note: Check layer normalization
        self.ln_final = LayerNorm(self.cfg)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.n_classes)
        self.setup()






    # def __init__(self, config, logger=None):
    #     super(BaseViT, self).__init__()

    #     self.logger = logger
    #     self.config = config

    #     self.config.transformer.activation_fn = activation_dict[self.config.transformer.activation_name]

    #     layer_norm = self.config.layernorm.layer_norm_eps
    #     hidden_dim = self.config.transformer.hidden_dim

    #     self.patch_embedding = self.config.embed_fn() if hasattr(self.config, 'embed_fn') else PatchEmbedding(config)
    #     self.patch_dropout = nn.Dropout(self.config.dropout.patch)

    #     self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if not self.config.classification.global_pool else None
    #     num_patches = (self.config.image.image_size // self.config.image.patch_size)**2
    #     token_length = num_patches + 1 if not self.config.classification.global_pool else num_patches
    #     self.position_embedding = nn.Parameter(torch.randn(1, token_length, hidden_dim))
    #     self.position_dropout = nn.Dropout(self.config.dropout.position)
    #     self.pre_block_norm = nn.LayerNorm(hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()
    #     block_fn = self.config.transformer.block_fn if hasattr(self.config.transformer, 'block_fn') else TransformerBlock
    #     self.blocks = nn.Sequential(*[block_fn(self.config) for _ in range(self.config.transformer.num_layers)])
    #     self.pre_head_norm = nn.LayerNorm(hidden_dim, eps=layer_norm) if layer_norm > 0 else nn.Identity()
    #     self.head = nn.Linear(hidden_dim, self.config.classification.num_classes)

    #     self.init_weights()

    # def init_weights(self):
    #     if not self.config.classification.global_pool:
    #         nn.init.normal_(self.cls_token, std=self.config.init.cls_std)
    #     nn.init.trunc_normal_(self.position_embedding, std=self.config.init.pos_std)   
    #     if self.config.init.weight_type == 'he':
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 nn.init.kaiming_normal_(m.weight, nonlinearity=initialization_dict[self.config.transformer.activation_name])
    #                 if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)

    # def forward(self, x, pre_logits: bool = False):
    #     x = self.patch_embedding(x)
    #     x = self.patch_dropout(x) if self.config.dropout.patch > 0 else x
    #     if not self.config.classification.global_pool:
    #         x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     x = x + self.position_embedding
    #     x = self.position_dropout(x) if self.config.dropout.position > 0 else x
    #     x = self.pre_block_norm(x)
    #     x = self.blocks(x)
    #     if self.config.classification.global_pool:  # GAAP
    #         x = x.mean(dim=1)
    #     else:  # CLS token
    #         x = x[:, 0]
    #     x = self.pre_head_norm(x)
    #     return x if pre_logits else self.head(x)

