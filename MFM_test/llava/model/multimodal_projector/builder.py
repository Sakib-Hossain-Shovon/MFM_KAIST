import torch
import torch.nn as nn
import re
from .llamagen_decoder import LlamaGenDecoderWrapper  #new


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print("DEBUG mm_projector_type =", projector_type)   #added

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    # ------------------------------------------------
# # LlamaGen vision decoder  #new
# # ------------------------------------------------
#     if projector_type == "llamagen_decoder":
#         return LlamaGenDecoderWrapper(
#             mm_hidden_size=config.mm_hidden_size,
#             decoder_hidden_size=config.mm_hidden_size,
#             pretrained_path=getattr(config, "llamagen_ckpt", None),
#             use_residual=True
#         )

#     raise ValueError(f'Unknown projector type: {projector_type}')
    
    # LlamaGen vision decoder
# ------------------------------------------------
    if projector_type == "llamagen_decoder":
        print("DEBUG: builder creating llamagen_decoder")
        return LlamaGenDecoderWrapper(
            mm_hidden_size=config.mm_hidden_size,
            decoder_hidden_size=config.mm_hidden_size,
            pretrained_path=getattr(config, "llamagen_ckpt", None),
            use_residual=True,

            llamagen_repo_path=getattr(config, "llamagen_repo_path", None),
            auto_download=getattr(config, "llamagen_auto_download", False),
            repo_id=getattr(config, "llamagen_repo_id", "peizesun/llamagen_t2i"),
            vq_ckpt_name=getattr(config, "llamagen_vq_ckpt_name", "vq_ds16_t2i.pt"),
            gpt_ckpt_name=getattr(config, "llamagen_gpt_ckpt_name", "t2i_XL_stage2_512.pt"),
            gpt_model_name=getattr(config, "llamagen_gpt_model_name", "GPT-XL"),
            vq_model_name=getattr(config, "llamagen_vq_model_name", "VQ-16"),
        )

    raise ValueError(f'Unknown projector type: {projector_type}')
