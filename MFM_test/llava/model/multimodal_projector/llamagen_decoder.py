# # import os
# # import sys
# # import importlib
# # from typing import Optional, Tuple

# # import torch
# # import torch.nn as nn


# # class LlamaGenDecoderWrapper(nn.Module):
# #     """
# #     A safe integration wrapper for plugging a pretrained LlamaGen-style image decoder
# #     between the CLIP vision encoder and the LLaVA multimodal projector.

# #     Target usage in LLaVA:
# #         vision_features = vision_tower(images)
# #         refined_features = image_decoder(vision_features)
# #         projected_features = mm_projector(refined_features)

# #     IMPORTANT:
# #     ----------
# #     This wrapper is intentionally conservative. Public LlamaGen docs show released
# #     VQ-VAE/tokenizer checkpoints and AR GPT checkpoints, but not a stable embeddable
# #     Python API for "drop-in decoder refinement" inside LLaVA. So this file gives you:

# #     1) a ready wrapper structure
# #     2) shape-safe adapters
# #     3) freeze/eval handling
# #     4) a single forward API for your LLaVA integration

# #     You still need to connect the exact LlamaGen repo classes on your machine.
# #     """

# #     def __init__(
# #         self,
# #         mm_hidden_size: int,
# #         hidden_size: Optional[int] = None,
# #         llamagen_repo_path: Optional[str] = None,
# #         vq_ckpt_path: Optional[str] = None,
# #         gpt_ckpt_path: Optional[str] = None,
# #         gpt_model_name: str = "GPT-XL",
# #         image_size: int = 256,
# #         device: Optional[torch.device] = None,
# #         freeze_pretrained: bool = True,
# #         use_residual: bool = True,
# #         adapter_ratio: float = 1.0,
# #     ):
# #         super().__init__()

# #         self.mm_hidden_size = mm_hidden_size
# #         self.hidden_size = hidden_size if hidden_size is not None else mm_hidden_size
# #         self.llamagen_repo_path = llamagen_repo_path
# #         self.vq_ckpt_path = vq_ckpt_path
# #         self.gpt_ckpt_path = gpt_ckpt_path
# #         self.gpt_model_name = gpt_model_name
# #         self.image_size = image_size
# #         self.freeze_pretrained = freeze_pretrained
# #         self.use_residual = use_residual
# #         self.device_override = device

# #         # ------------------------------------------------------------
# #         # Adapters:
# #         # CLIP/LLaVA visual features -> decoder latent space -> back
# #         # ------------------------------------------------------------
# #         adapter_hidden = max(64, int(self.mm_hidden_size * adapter_ratio))

# #         self.in_norm = nn.LayerNorm(self.mm_hidden_size)
# #         self.in_proj = nn.Sequential(
# #             nn.Linear(self.mm_hidden_size, adapter_hidden),
# #             nn.GELU(),
# #             nn.Linear(adapter_hidden, self.mm_hidden_size),
# #         )

# #         self.out_norm = nn.LayerNorm(self.mm_hidden_size)
# #         self.out_proj = nn.Sequential(
# #             nn.Linear(self.mm_hidden_size, adapter_hidden),
# #             nn.GELU(),
# #             nn.Linear(adapter_hidden, self.mm_hidden_size),
# #         )

# #         # Fallback refinement block so the module still has a valid forward path
# #         # even before exact LlamaGen internals are wired in.
# #         self.fallback_refiner = nn.Sequential(
# #             nn.LayerNorm(self.mm_hidden_size),
# #             nn.Linear(self.mm_hidden_size, self.mm_hidden_size),
# #             nn.GELU(),
# #             nn.Linear(self.mm_hidden_size, self.mm_hidden_size),
# #         )

# #         # ------------------------------------------------------------
# #         # Placeholders for actual LlamaGen modules
# #         # ------------------------------------------------------------
# #         self.vq_model = None
# #         self.gpt_model = None
# #         self.has_pretrained_llamagen = False

# #         # Try to load external LlamaGen repo/classes if provided
# #         self._try_load_llamagen()

# #         if self.freeze_pretrained:
# #             self.freeze_pretrained_modules()

# #     def _append_repo_to_syspath(self):
# #         if self.llamagen_repo_path is None:
# #             return
# #         repo_path = os.path.abspath(self.llamagen_repo_path)
# #         if repo_path not in sys.path:
# #             sys.path.insert(0, repo_path)

# #     def _safe_torch_load(self, ckpt_path: str):
# #         if ckpt_path is None:
# #             return None
# #         if not os.path.exists(ckpt_path):
# #             raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
# #         return torch.load(ckpt_path, map_location="cpu")

# #     def _try_load_llamagen(self):
# #         """
# #         Best-effort loader.

# #         Because LlamaGen's public docs expose checkpoints and shell usage, but not a
# #         stable embedding API for external projects, this function is written to be
# #         edited in one place once you clone the repo locally.

# #         You will need to adjust the import targets below to match your installed
# #         LlamaGen repo.
# #         """
# #         try:
# #             self._append_repo_to_syspath()

# #             # ------------------------------------------------------------------
# #             # TODO: EDIT THESE IMPORTS TO MATCH YOUR LOCAL LLAMAGEN REPO
# #             #
# #             # Common pattern after cloning FoundationVision/LlamaGen:
# #             #   repo/
# #             #     autoregressive/
# #             #     tokenizer/
# #             #
# #             # You need to replace these with the real class factories from your
# #             # local checked-out repo.
# #             # ------------------------------------------------------------------

# #             # Example placeholders:
# #             # gpt_module = importlib.import_module("autoregressive.models.gpt")
# #             # vq_module = importlib.import_module("tokenizer.models.vq_model")

# #             # ------------------------------------------------------------------
# #             # Since public docs do not guarantee these exact import paths, we keep
# #             # the wrapper safe by default and do not hard-fail here.
# #             # ------------------------------------------------------------------

# #             if self.vq_ckpt_path is None or self.gpt_ckpt_path is None:
# #                 return

# #             # Load checkpoint dicts now so failures happen early
# #             _ = self._safe_torch_load(self.vq_ckpt_path)
# #             _ = self._safe_torch_load(self.gpt_ckpt_path)

# #             # ------------------------------------------------------------------
# #             # IMPORTANT:
# #             # After you identify the exact LlamaGen classes in your cloned repo,
# #             # replace the return below by actual model construction + load_state_dict.
# #             #
# #             # Example pattern:
# #             #   self.vq_model = build_vq_model(...)
# #             #   self.gpt_model = build_gpt_model(...)
# #             #   self.vq_model.load_state_dict(vq_state, strict=False)
# #             #   self.gpt_model.load_state_dict(gpt_state, strict=False)
# #             #   self.vq_model.eval()
# #             #   self.gpt_model.eval()
# #             #   self.has_pretrained_llamagen = True
# #             # ------------------------------------------------------------------

# #             return

# #         except Exception as e:
# #             print(f"[LlamaGenDecoderWrapper] Warning: pretrained LlamaGen not loaded. Reason: {e}")
# #             self.vq_model = None
# #             self.gpt_model = None
# #             self.has_pretrained_llamagen = False

# #     def freeze_pretrained_modules(self):
# #         for module in [self.vq_model, self.gpt_model]:
# #             if module is not None:
# #                 module.eval()
# #                 for p in module.parameters():
# #                     p.requires_grad = False

# #     def _get_target_device_dtype(self, x: torch.Tensor) -> Tuple[torch.device, torch.dtype]:
# #         device = self.device_override if self.device_override is not None else x.device
# #         dtype = x.dtype
# #         return device, dtype

# #     def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """
# #         Safe fallback path before exact pretrained LlamaGen internals are wired in.
# #         Keeps tensor shapes unchanged.
# #         """
# #         h = self.in_norm(x)
# #         h = self.in_proj(h)
# #         h = self.fallback_refiner(h)
# #         h = self.out_proj(h)
# #         h = self.out_norm(h)
# #         if self.use_residual:
# #             h = h + x
# #         return h

# #     def _pretrained_llamagen_forward(self, x: torch.Tensor) -> torch.Tensor:
# #         """
# #         THIS IS THE ONLY METHOD YOU NEED TO EDIT after you confirm exact LlamaGen API.

# #         Expected input:
# #             x: [B, N, D]  (CLIP/LLaVA image features)

# #         Expected output:
# #             refined_x: [B, N, D]  (same shape, refined features)

# #         Practical guidance:
# #         -------------------
# #         Since LlamaGen is released as VQ-VAE/tokenizer + AR GPT checkpoints, a true
# #         integration usually needs:
# #             1) map x into a compatible latent/token space
# #             2) use pretrained LlamaGen components to refine/generate latent structure
# #             3) map back to [B, N, D]

# #         For now, until exact repo internals are confirmed, this method falls back
# #         to a shape-safe refinement path.
# #         """
# #         # ------------------------------------------------------------
# #         # Replace this whole body with the real LlamaGen call path later.
# #         # ------------------------------------------------------------
# #         return self._fallback_forward(x)

# #     def forward(self, image_features: torch.Tensor) -> torch.Tensor:
# #         """
# #         image_features: [B, N, D]
# #         returns:        [B, N, D]
# #         """
# #         if not isinstance(image_features, torch.Tensor):
# #             raise TypeError(f"Expected torch.Tensor, got {type(image_features)}")

# #         if image_features.ndim != 3:
# #             raise ValueError(
# #                 f"LlamaGenDecoderWrapper expects [B, N, D], got shape {tuple(image_features.shape)}"
# #             )

# #         device, dtype = self._get_target_device_dtype(image_features)
# #         x = image_features.to(device=device, dtype=dtype)

# #         if self.has_pretrained_llamagen:
# #             refined = self._pretrained_llamagen_forward(x)
# #         else:
# #             refined = self._fallback_forward(x)

# #         if refined.shape != image_features.shape:
# #             raise RuntimeError(
# #                 f"Decoder output shape must match input shape. "
# #                 f"Got input={tuple(image_features.shape)}, output={tuple(refined.shape)}"
# #             )

# #         return refined

# #     @property
# #     def config(self):
# #         return {
# #             "mm_projector_type": "llamagen_decoder",
# #             "mm_hidden_size": self.mm_hidden_size,
# #             "hidden_size": self.hidden_size,
# #             "has_pretrained_llamagen": self.has_pretrained_llamagen,
# #             "gpt_model_name": self.gpt_model_name,
# #             "image_size": self.image_size,
# #         }


# import sys

# llamagen_repo_path = "/data2/sakib/LlamaGen"
# if llamagen_repo_path not in sys.path:
#     sys.path.insert(0, llamagen_repo_path)

# # old: from tokenizer_image import ...
# # new:
# # from tokenizer.tokenizer_image import

# import os
# import sys
# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class LlamaGenDecoderWrapper(nn.Module):
#     """
#     Loads official pretrained LlamaGen checkpoints and provides a shape-safe
#     decoder-like refinement block for LLaVA image features.

#     IMPORTANT:
#     ----------
#     This file now truly loads official LlamaGen weights if:
#       1) you cloned the official LlamaGen repo locally, and
#       2) you provide (or auto-download) the official checkpoints.

#     However, LlamaGen is not natively designed to consume CLIP visual embeddings.
#     So the forward path below uses pretrained VQ codebook priors in a safe adapter,
#     rather than claiming a fully correct CLIP->LlamaGen token bridge.

#     Expected LLaVA input/output:
#       input  : [B, N, D]
#       output : [B, N, D]
#     """

#     def __init__(
#         self,
#         mm_hidden_size: int,
#         decoder_hidden_size: Optional[int] = None,
#         pretrained_path: Optional[str] = None,   # optional local VQ ckpt path shortcut
#         use_residual: bool = True,

#         # ---- real LlamaGen loading config ----
#         llamagen_repo_path: Optional[str] = None,
#         auto_download: bool = False,
#         repo_id: str = "peizesun/llamagen_t2i",
#         vq_ckpt_name: str = "vq_ds16_t2i.pt",
#         gpt_ckpt_name: str = "t2i_XL_stage2_512.pt",
#         gpt_model_name: str = "GPT-XL",
#         vq_model_name: str = "VQ-16",
#         codebook_size: int = 16384,
#         codebook_embed_dim: int = 8,
#         downsample_size: int = 16,
#         image_size: int = 512,
#         num_classes: int = 1000,
#         cls_token_num: int = 120,
#         gpt_type: str = "t2i",
#         precision: str = "fp16",
#         from_fsdp: bool = False,
#         strict_pretrained: bool = False,
#         freeze_pretrained: bool = True,
#     ):
#         super().__init__()
#         print("DEBUG: LlamaGenDecoderWrapper init called")

#         self.mm_hidden_size = mm_hidden_size
#         self.decoder_hidden_size = decoder_hidden_size or mm_hidden_size
#         self.use_residual = use_residual

#         self.llamagen_repo_path = llamagen_repo_path
#         self.auto_download = auto_download
#         self.repo_id = repo_id
#         self.vq_ckpt_name = pretrained_path if pretrained_path is not None else vq_ckpt_name
#         self.gpt_ckpt_name = gpt_ckpt_name
#         self.gpt_model_name = gpt_model_name
#         self.vq_model_name = vq_model_name
#         self.codebook_size = codebook_size
#         self.codebook_embed_dim = codebook_embed_dim
#         self.downsample_size = downsample_size
#         self.image_size = image_size
#         self.num_classes = num_classes
#         self.cls_token_num = cls_token_num
#         self.gpt_type = gpt_type
#         self.precision = precision
#         self.from_fsdp = from_fsdp
#         self.strict_pretrained = strict_pretrained
#         self.freeze_pretrained = freeze_pretrained

#         # ------------------------------------------------------------
#         # Shape-safe adapters for LLaVA feature space
#         # ------------------------------------------------------------
#         self.input_norm = nn.LayerNorm(self.mm_hidden_size)
#         self.input_proj = nn.Sequential(
#             nn.Linear(self.mm_hidden_size, self.decoder_hidden_size),
#             nn.GELU(),
#             nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
#         )

#         self.output_proj = nn.Sequential(
#             nn.Linear(self.decoder_hidden_size, self.mm_hidden_size),
#             nn.GELU(),
#             nn.Linear(self.mm_hidden_size, self.mm_hidden_size),
#         )
#         self.output_norm = nn.LayerNorm(self.mm_hidden_size)

#         # ------------------------------------------------------------
#         # Fallback local refiner (used if true pretrained load is absent)
#         # ------------------------------------------------------------
#         self.fallback_refiner = nn.Sequential(
#             nn.LayerNorm(self.decoder_hidden_size),
#             nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
#             nn.GELU(),
#             nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
#         )

#         # ------------------------------------------------------------
#         # Official LlamaGen modules
#         # ------------------------------------------------------------
#         self.vq_model = None
#         self.gpt_model = None
#         self.vq_codebook_proj_in = None
#         self.vq_codebook_proj_out = None
#         self.has_real_pretrained = False

#         self._try_load_official_llamagen()

#         if self.freeze_pretrained:
#             self._freeze_pretrained()

#     def _append_repo_to_syspath(self):
#         if self.llamagen_repo_path is None:
#             return
#         repo_path = os.path.abspath(self.llamagen_repo_path)
#         if repo_path not in sys.path:
#             sys.path.insert(0, repo_path)

#     def _resolve_ckpt_path(self, ckpt_name_or_path: str) -> str:
#         """
#         Accept either:
#           - an absolute/local checkpoint path
#           - a filename that should be auto-downloaded from HF repo
#         """
#         if ckpt_name_or_path is None:
#             return None

#         if os.path.exists(ckpt_name_or_path):
#             return ckpt_name_or_path

#         if not self.auto_download:
#             raise FileNotFoundError(
#                 f"Checkpoint not found locally: {ckpt_name_or_path}\n"
#                 f"Either provide a valid local path or set auto_download=True."
#             )

#         from huggingface_hub import hf_hub_download
#         local_path = hf_hub_download(repo_id=self.repo_id, filename=ckpt_name_or_path)
#         return local_path

#     def _get_checkpoint_state_dict_for_gpt(self, checkpoint: dict) -> dict:
#         """
#         Mirrors official LlamaGen loading logic used in the demo/app.
#         """
#         if self.from_fsdp:
#             return checkpoint
#         if "model" in checkpoint:
#             return checkpoint["model"]
#         if "module" in checkpoint:
#             return checkpoint["module"]
#         if "state_dict" in checkpoint:
#             return checkpoint["state_dict"]
#         raise RuntimeError("Unsupported GPT checkpoint format. Please inspect the checkpoint.")

#     def _try_load_official_llamagen(self):
#         print("DEBUG: trying to load official LlamaGen")
        
#         try:
#             self._append_repo_to_syspath()

        

#             # Official import structure shown in LlamaGen app/demo
#             # from tokenizer_image.vq_model import VQ_models
#             # from models.gpt import GPT_models

#         #again_new#############
#             from tokenizer.tokenizer_image.vq_model import VQ_models
#             from autoregressive.models.gpt import GPT_models



#             ###############
            
#             # imports above are consistent with the public demo/app. 
#             # See official HF space lines showing:
#             # from tokenizer_image.vq_model import VQ_models
#             # from models.gpt import GPT_models

#             vq_ckpt_path = self._resolve_ckpt_path(self.vq_ckpt_name)
#             gpt_ckpt_path = self._resolve_ckpt_path(self.gpt_ckpt_name)

#             # ---- create VQ model ----
#             self.vq_model = VQ_models[self.vq_model_name](
#                 codebook_size=self.codebook_size,
#                 codebook_embed_dim=self.codebook_embed_dim,
#             )

#             vq_ckpt = torch.load(vq_ckpt_path, map_location="cpu")
#             if "model" not in vq_ckpt:
#                 raise RuntimeError("Unexpected VQ checkpoint format: missing key 'model'.")
#             self.vq_model.load_state_dict(vq_ckpt["model"], strict=False)
#             self.vq_model.eval()

#             # ---- create GPT model ----
#             latent_size = self.image_size // self.downsample_size
#             gpt_dtype = {
#                 "none": torch.float32,
#                 "fp16": torch.float16,
#                 "bf16": torch.bfloat16,
#             }[self.precision]

#             self.gpt_model = GPT_models[self.gpt_model_name](
#                 vocab_size=self.codebook_size,
#                 block_size=latent_size ** 2,
#                 num_classes=self.num_classes,
#                 cls_token_num=self.cls_token_num,
#                 model_type=self.gpt_type,
#             ).to(dtype=gpt_dtype)

#             gpt_ckpt = torch.load(gpt_ckpt_path, map_location="cpu")
#             gpt_state = self._get_checkpoint_state_dict_for_gpt(gpt_ckpt)
#             self.gpt_model.load_state_dict(gpt_state, strict=False)
#             self.gpt_model.eval()

#             # ---- use pretrained VQ codebook as a real prior ----
#             codebook_weight = self._extract_vq_codebook_weight(self.vq_model)
#             if codebook_weight is None:
#                 raise RuntimeError(
#                     "Could not find VQ codebook embeddings inside the loaded VQ model. "
#                     "Please inspect your local LlamaGen VQ model structure."
#                 )

#             codebook_dim = codebook_weight.shape[1]

#             self.vq_codebook_proj_in = nn.Linear(self.decoder_hidden_size, codebook_dim)
#             self.vq_codebook_proj_out = nn.Linear(codebook_dim, self.decoder_hidden_size)

#             self.register_buffer("_vq_codebook_weight", codebook_weight.detach().clone(), persistent=False)

#             self.has_real_pretrained = True
#             print("[LlamaGenDecoder] Official pretrained VQ/GPT checkpoints loaded successfully.")


#             ##new ###########
#             print("[LlamaGenDecoder] Official pretrained VQ/GPT checkpoints loaded successfully.")

#             print(
#                 "CHECK after official load input_norm:",
#                 "has_nan=", torch.isnan(self.input_norm.weight).any().item(),
#                 "has_inf=", torch.isinf(self.input_norm.weight).any().item(),
#                 "min=", self.input_norm.weight.min().item(),
#                 "max=", self.input_norm.weight.max().item(),
#             )
# #############

#         except Exception as e:
#             self.has_real_pretrained = False
#             self.vq_model = None
#             self.gpt_model = None
#             print(f"[LlamaGenDecoder] Warning: failed to load official pretrained LlamaGen modules: {e}")
#             if self.strict_pretrained:
#                 raise

#     def _extract_vq_codebook_weight(self, vq_model):
#         """
#         Best-effort extraction of the VQ codebook embedding matrix.
#         Different repos may store it under slightly different names.
#         """
#         # Common possibilities
#         candidates = [
#             "quantize.embedding.weight",
#             "quantize.embed.weight",
#             "embedding.weight",
#             "codebook.weight",
#         ]

#         state = vq_model.state_dict()
#         for k in candidates:
#             if k in state:
#                 return state[k]

#         # fallback scan
#         for k, v in state.items():
#             if ("embed" in k or "codebook" in k or "embedding" in k) and v.ndim == 2:
#                 # prefer [codebook_size, embed_dim]
#                 if v.shape[0] == self.codebook_size:
#                     return v

#         return None

#     def _freeze_pretrained(self):
#         for module in [self.vq_model, self.gpt_model]:
#             if module is not None:
#                 module.eval()
#                 for p in module.parameters():
#                     p.requires_grad = False

#     def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.input_norm(x)
#         h = self.input_proj(h)
#         h = self.fallback_refiner(h)
#         h = self.output_proj(h)
#         h = self.output_norm(h)
#         if self.use_residual:
#             h = h + x
#         return h

#     def _pretrained_codebook_refine(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Uses the real pretrained VQ codebook as a frozen latent prior.

#         This is NOT a full official CLIP->LlamaGen generation path.
#         It is a safe way to inject *real pretrained LlamaGen weights* into the
#         refinement process without pretending that CLIP features are native
#         LlamaGen tokens.
#         """
#         h = self.input_norm(x)

#         #new################
#         print(
#             "CHECK init input_norm:",
#             "has_nan=", torch.isnan(self.input_norm.weight).any().item(),
#             "has_inf=", torch.isinf(self.input_norm.weight).any().item(),
#             "min=", self.input_norm.weight.min().item(),
#             "max=", self.input_norm.weight.max().item(),
# )
#         ################
#         h = self.input_proj(h)  # [B, N, D_dec]

#         # map to VQ codebook embedding space
#         z = self.vq_codebook_proj_in(h)  # [B, N, C]
#         codebook = self._vq_codebook_weight.to(device=z.device, dtype=z.dtype)  # [K, C]

#         # soft assignment over pretrained codebook
#         logits = torch.matmul(z, codebook.t()) / max(codebook.shape[1] ** 0.5, 1.0)  # [B, N, K]
#         probs = F.softmax(logits, dim=-1)

#         # reconstruct a codebook-conditioned latent
#         z_refined = torch.matmul(probs, codebook)  # [B, N, C]
#         h_refined = self.vq_codebook_proj_out(z_refined)  # [B, N, D_dec]

#         # light local refiner on top
#         h_refined = h_refined + self.fallback_refiner(h)

#         y = self.output_proj(h_refined)
#         y = self.output_norm(y)

#         if self.use_residual:
#             y = y + x

#         return y

#     def forward(self, image_features: torch.Tensor) -> torch.Tensor:
#         if not isinstance(image_features, torch.Tensor):
#             raise TypeError(f"Expected torch.Tensor, got {type(image_features)}")

#         if image_features.ndim != 3:
#             raise ValueError(
#                 f"LlamaGenDecoderWrapper expects [B, N, D], got shape {tuple(image_features.shape)}"
#             )

#         if self.has_real_pretrained and self.vq_codebook_proj_in is not None:
#             out = self._pretrained_codebook_refine(image_features)
#         else:
#             out = self._fallback_forward(image_features)

#         if out.shape != image_features.shape:
#             raise RuntimeError(
#                 f"Decoder output shape mismatch: input={tuple(image_features.shape)}, output={tuple(out.shape)}"
#             )

#         return out

#     @property
#     def config(self):
#         return {
#             "mm_projector_type": "llamagen_decoder",
#             "mm_hidden_size": self.mm_hidden_size,
#             "decoder_hidden_size": self.decoder_hidden_size,
#             "has_real_pretrained": self.has_real_pretrained,
#             "repo_id": self.repo_id,
#             "vq_ckpt_name": self.vq_ckpt_name,
#             "gpt_ckpt_name": self.gpt_ckpt_name,
#             "gpt_model_name": self.gpt_model_name,
#             "vq_model_name": self.vq_model_name,
#         }


######Md Sakib Hossain#######

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaGenDecoderWrapper(nn.Module):
    """
    Shape-safe wrapper that injects pretrained LlamaGen priors into a refinement
    block for LLaVA image features.

    Expected input/output:
        input  : [B, N, D]
        output : [B, N, D]

    Notes
    -----
    - This is NOT a native CLIP -> LlamaGen generation path.
    - It uses the official pretrained LlamaGen VQ/GPT components as priors where possible.
    - Wrapper-side modules (LayerNorm / projections / fallback refiner) are local adapter modules.
    """

    def __init__(
        self,
        mm_hidden_size: int,
        decoder_hidden_size: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        use_residual: bool = True,

        # official LlamaGen config
        llamagen_repo_path: Optional[str] = None,
        auto_download: bool = False,
        repo_id: str = "peizesun/llamagen_t2i",
        vq_ckpt_name: str = "vq_ds16_t2i.pt",
        gpt_ckpt_name: str = "t2i_XL_stage2_512.pt",
        gpt_model_name: str = "GPT-XL",
        vq_model_name: str = "VQ-16",
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        downsample_size: int = 16,
        image_size: int = 512,
        num_classes: int = 1000,
        cls_token_num: int = 120,
        gpt_type: str = "t2i",
        precision: str = "fp16",
        from_fsdp: bool = False,
        strict_pretrained: bool = False,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
    
        #####new###
        print("DEBUG: LlamaGenDecoderWrapper init called FROM:", __file__)
        print("DEBUG auto_download =", auto_download, "repo_path =", llamagen_repo_path)
###

        self.mm_hidden_size = mm_hidden_size
        self.decoder_hidden_size = decoder_hidden_size or mm_hidden_size
        self.use_residual = use_residual

        self.llamagen_repo_path = llamagen_repo_path
        self.auto_download = auto_download
        self.repo_id = repo_id
        self.vq_ckpt_name = pretrained_path if pretrained_path is not None else vq_ckpt_name
        self.gpt_ckpt_name = gpt_ckpt_name
        self.gpt_model_name = gpt_model_name
        self.vq_model_name = vq_model_name
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.downsample_size = downsample_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.cls_token_num = cls_token_num
        self.gpt_type = gpt_type
        self.precision = precision
        self.from_fsdp = from_fsdp
        self.strict_pretrained = strict_pretrained
        self.freeze_pretrained = freeze_pretrained

        # -----------------------------
        # Local wrapper-side adapters
        # -----------------------------
        self.input_norm = nn.LayerNorm(self.mm_hidden_size)
        self.input_proj = nn.Sequential(
            nn.Linear(self.mm_hidden_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.decoder_hidden_size, self.mm_hidden_size),
            nn.GELU(),
            nn.Linear(self.mm_hidden_size, self.mm_hidden_size),
        )
        self.output_norm = nn.LayerNorm(self.mm_hidden_size)

        self.fallback_refiner = nn.Sequential(
            nn.LayerNorm(self.decoder_hidden_size),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
        )

        # -----------------------------
        # Official pretrained modules
        # -----------------------------
        self.vq_model = None
        self.gpt_model = None
        self.vq_codebook_proj_in = None
        self.vq_codebook_proj_out = None
        # self._vq_codebook_weight = None
        self.register_buffer("_vq_codebook_weight", None, persistent=False)
        self.has_real_pretrained = False

        # init-time check
        self._print_param_health("CHECK init input_norm", self.input_norm.weight)

        self._try_load_official_llamagen()

        if self.freeze_pretrained:
            self._freeze_pretrained()

        # end-of-init check
        self._print_param_health("CHECK end __init__ input_norm", self.input_norm.weight)

    # ------------------------------------------------------------------
    # Utility / diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _print_param_health(tag: str, tensor: torch.Tensor):
        with torch.no_grad():
            has_finite = torch.isfinite(tensor).any().item()
            print(
                f"{tag}:",
                "shape=", tuple(tensor.shape),
                "dtype=", tensor.dtype,
                "has_nan=", torch.isnan(tensor).any().item(),
                "has_inf=", torch.isinf(tensor).any().item(),
                "min=", tensor.min().item() if has_finite else "no_finite",
                "max=", tensor.max().item() if has_finite else "no_finite",
            )

    def _append_repo_to_syspath(self):
        if self.llamagen_repo_path is None:
            return
        repo_path = os.path.abspath(self.llamagen_repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    
    def _freeze_pretrained(self):
        for module in [self.vq_model, self.gpt_model]:
            if module is not None:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False


    def _resolve_ckpt_path(self, ckpt_name_or_path: str) -> str:
        if ckpt_name_or_path is None:
            return None

        if os.path.exists(ckpt_name_or_path):
            return ckpt_name_or_path

        if not self.auto_download:
            raise FileNotFoundError(
                f"Checkpoint not found locally: {ckpt_name_or_path}. "
                f"Either provide a valid local path or set auto_download=True."
            )

        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self.repo_id, filename=ckpt_name_or_path)


    def _try_load_official_llamagen(self):
        try:
            print("DEBUG: trying to load official LlamaGen")
            self._append_repo_to_syspath()

            # import official modules lazily
            from tokenizer.tokenizer_image.vq_model import VQ_models
            from autoregressive.models.gpt import GPT_models

            # build official VQ model
            if self.vq_model_name not in VQ_models:
                raise ValueError(f"Unknown VQ model: {self.vq_model_name}")
            self.vq_model = VQ_models[self.vq_model_name](
                codebook_size=self.codebook_size,
                codebook_embed_dim=self.codebook_embed_dim,
            )

            # build official GPT model
            if self.gpt_model_name not in GPT_models:
                raise ValueError(f"Unknown GPT model: {self.gpt_model_name}")
            latent_size = self.image_size // self.downsample_size
            self.gpt_model = GPT_models[self.gpt_model_name](
                block_size=latent_size ** 2,
                cls_token_num=self.cls_token_num,
                model_type=self.gpt_type,
                vocab_size=self.codebook_size,
                num_classes=self.num_classes,
            )

            # resolve/download checkpoints
            vq_ckpt_path = self._resolve_ckpt_path(self.vq_ckpt_name)
            gpt_ckpt_path = self._resolve_ckpt_path(self.gpt_ckpt_name)

            # load VQ checkpoint
            vq_ckpt = torch.load(vq_ckpt_path, map_location="cpu")
            vq_state = vq_ckpt["model"] if isinstance(vq_ckpt, dict) and "model" in vq_ckpt else vq_ckpt
            self.vq_model.load_state_dict(vq_state, strict=False)

            # load GPT checkpoint
            gpt_ckpt = torch.load(gpt_ckpt_path, map_location="cpu")
            gpt_state = gpt_ckpt["model"] if isinstance(gpt_ckpt, dict) and "model" in gpt_ckpt else gpt_ckpt
            self.gpt_model.load_state_dict(gpt_state, strict=False)

            # retrieve VQ codebook
            codebook = self._extract_vq_codebook()
            if codebook is None:
                raise RuntimeError("Failed to locate VQ codebook embedding in official VQ model.")

            self.register_buffer("_vq_codebook_weight", codebook.detach().float(), persistent=False)

            # wrapper-side projections between adapter hidden dim and codebook embed dim
            self.vq_codebook_proj_in = nn.Linear(self.decoder_hidden_size, self.codebook_embed_dim)
            self.vq_codebook_proj_out = nn.Linear(self.codebook_embed_dim, self.decoder_hidden_size)

            self.has_real_pretrained = True

            print("[LlamaGenDecoder] Official pretrained VQ/GPT checkpoints loaded successfully.")
            self._print_param_health("CHECK after official load input_norm", self.input_norm.weight)

        except Exception as e:
            self.vq_model = None
            self.gpt_model = None
            self.vq_codebook_proj_in = None
            self.vq_codebook_proj_out = None
            # self._vq_codebook_weight = None
            self.register_buffer("_vq_codebook_weight", None, persistent=False)

            self.has_real_pretrained = False
            print(f"[LlamaGenDecoder] Official load failed, using fallback path. Reason: {e}")

    # def _append_repo_to_syspath(self):
    #     if self.llamagen_repo_path is None:
    #         return
    #     repo_path = os.path.abspath(self.llamagen_repo_path)
    #     if repo_path not in sys.path:
    #         sys.path.insert(0, repo_path)

    # def _freeze_pretrained(self):
    #     for module in [self.vq_model, self.gpt_model]:
    #         if module is not None:
    #             module.eval()
    #             for p in module.parameters():
    #                 p.requires_grad = False

    # # ------------------------------------------------------------------
    # # Official LlamaGen loading
    # # ------------------------------------------------------------------
    # def _try_load_official_llamagen(self):
    #     try:
    #         print("DEBUG: trying to load official LlamaGen")
    #         self._append_repo_to_syspath()

    #         # import official modules lazily
    #         from tokenizer.tokenizer_image.vq_model import VQ_models
    #         from autoregressive.models.gpt import GPT_models

    #         # build official VQ model
    #         if self.vq_model_name not in VQ_models:
    #             raise ValueError(f"Unknown VQ model: {self.vq_model_name}")
    #         self.vq_model = VQ_models[self.vq_model_name](
    #             codebook_size=self.codebook_size,
    #             codebook_embed_dim=self.codebook_embed_dim,
    #         )

    #         # build official GPT model
    #         if self.gpt_model_name not in GPT_models:
    #             raise ValueError(f"Unknown GPT model: {self.gpt_model_name}")
    #         latent_size = self.image_size // self.downsample_size
    #         self.gpt_model = GPT_models[self.gpt_model_name](
    #             block_size=latent_size ** 2,
    #             cls_token_num=self.cls_token_num,
    #             model_type=self.gpt_type,
    #             vocab_size=self.codebook_size,
    #             num_classes=self.num_classes,
    #         )

    #         # load VQ checkpoint
    #         vq_ckpt_path = self.vq_ckpt_name
    #         if not os.path.isfile(vq_ckpt_path):
    #             raise FileNotFoundError(f"VQ checkpoint not found: {vq_ckpt_path}")
    #         vq_ckpt = torch.load(vq_ckpt_path, map_location="cpu")
    #         vq_state = vq_ckpt["model"] if isinstance(vq_ckpt, dict) and "model" in vq_ckpt else vq_ckpt
    #         self.vq_model.load_state_dict(vq_state, strict=False)

    #         # load GPT checkpoint
    #         gpt_ckpt_path = self.gpt_ckpt_name
    #         if not os.path.isfile(gpt_ckpt_path):
    #             raise FileNotFoundError(f"GPT checkpoint not found: {gpt_ckpt_path}")
    #         gpt_ckpt = torch.load(gpt_ckpt_path, map_location="cpu")
    #         gpt_state = gpt_ckpt["model"] if isinstance(gpt_ckpt, dict) and "model" in gpt_ckpt else gpt_ckpt
    #         self.gpt_model.load_state_dict(gpt_state, strict=False)

    #         # retrieve VQ codebook
    #         codebook = self._extract_vq_codebook()
    #         if codebook is None:
    #             raise RuntimeError("Failed to locate VQ codebook embedding in official VQ model.")

    #         self.register_buffer("_vq_codebook_weight", codebook.detach().float(), persistent=False)

    #         # wrapper-side projections between adapter hidden dim and codebook embed dim
    #         self.vq_codebook_proj_in = nn.Linear(self.decoder_hidden_size, self.codebook_embed_dim)
    #         self.vq_codebook_proj_out = nn.Linear(self.codebook_embed_dim, self.decoder_hidden_size)

    #         self.has_real_pretrained = True

    #         print("[LlamaGenDecoder] Official pretrained VQ/GPT checkpoints loaded successfully.")
    #         self._print_param_health("CHECK after official load input_norm", self.input_norm.weight)

    #     except Exception as e:
    #         self.vq_model = None
    #         self.gpt_model = None
    #         self.vq_codebook_proj_in = None
    #         self.vq_codebook_proj_out = None
    #         self._vq_codebook_weight = None
    #         self.has_real_pretrained = False
    #         print(f"[LlamaGenDecoder] Official load failed, using fallback path. Reason: {e}")

    def _extract_vq_codebook(self):
        if self.vq_model is None:
            return None

        state = self.vq_model.state_dict()
        candidates = [
            "quantize.embedding.weight",
            "quantize.embed.weight",
            "embedding.weight",
            "codebook.weight",
        ]

        for k in candidates:
            if k in state:
                return state[k]

        for k, v in state.items():
            if ("embed" in k or "codebook" in k or "embedding" in k) and v.ndim == 2:
                if v.shape[0] == self.codebook_size:
                    return v

        return None

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = self.fallback_refiner(h)
        h = self.output_proj(h)
        h = self.output_norm(h)
        if self.use_residual:
            h = h + x
        return h

    def _pretrained_codebook_refine(self, x: torch.Tensor) -> torch.Tensor:
        """
        Uses the pretrained VQ codebook as a frozen latent prior.
        This is a codebook-guided refinement path, not a native full LlamaGen decode.
        """
        h = self.input_norm(x)
        h = self.input_proj(h)  # [B, N, D_dec]

        z = self.vq_codebook_proj_in(h)  # [B, N, C]
        codebook = self._vq_codebook_weight.to(device=z.device, dtype=z.dtype)  # [K, C]

        logits = torch.matmul(z, codebook.t()) / max(codebook.shape[1] ** 0.5, 1.0)  # [B, N, K]
        probs = F.softmax(logits, dim=-1)

        z_refined = torch.matmul(probs, codebook)  # [B, N, C]
        h_refined = self.vq_codebook_proj_out(z_refined)  # [B, N, D_dec]

        h_refined = h_refined + self.fallback_refiner(h)

        y = self.output_proj(h_refined)
        y = self.output_norm(y)

        if self.use_residual:
            y = y + x

        return y

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(image_features, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image_features)}")

        if image_features.ndim != 3:
            raise ValueError(
                f"LlamaGenDecoderWrapper expects [B, N, D], got shape {tuple(image_features.shape)}"
            )

        if self.has_real_pretrained and self.vq_codebook_proj_in is not None:
            out = self._pretrained_codebook_refine(image_features)
        else:
            out = self._fallback_forward(image_features)

        if out.shape != image_features.shape:
            raise RuntimeError(
                f"Decoder output shape mismatch: input={tuple(image_features.shape)}, output={tuple(out.shape)}"
            )

        return out

    @property
    def config(self):
        return {
            "mm_projector_type": "llamagen_decoder",
            "mm_hidden_size": self.mm_hidden_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "has_real_pretrained": self.has_real_pretrained,
            "repo_id": self.repo_id,
            "vq_ckpt_name": self.vq_ckpt_name,
            "gpt_ckpt_name": self.gpt_ckpt_name,
            "gpt_model_name": self.gpt_model_name,
            "vq_model_name": self.vq_model_name,
        }