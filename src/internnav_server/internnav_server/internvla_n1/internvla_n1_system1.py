import os

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file

from .internvla_n1_arch import build_depthanythingv2
from .nextdit_traj import LuminaNextDiT2DModel

# Configuration constants
DIT_INPUT_SIZE = 8
DIT_PATCH_SIZE = 1
DIT_IN_CHANNELS = 384
DIT_HIDDEN_SIZE = 384
DIT_NUM_LAYERS = 12
DIT_NUM_HEADS = 6
DIT_LATENT_DIM = 768

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)
        return enc

class MemoryEncoder(nn.Module):
    def __init__(self, hidden_size=384, num_heads=6, num_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_pos = nn.Parameter(torch.randn(max_len, hidden_size))

    def forward(self, memory, memory_mask=None):
        B, N, C = memory.shape
        pos = self.memory_pos[:N, :].unsqueeze(0).expand(B, -1, -1)
        memory = memory + pos
        return self.encoder(memory, src_key_padding_mask=memory_mask)

class QFormer(nn.Module):
    def __init__(self, num_query=32, hidden_size=768, num_layers=3, num_heads=12):
        super().__init__()
        self.num_query = num_query
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.visual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_feats, visual_attn_mask=None):
        B = visual_feats.size(0)
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)
        return self.decoder(query_tokens, visual_feats, memory_key_padding_mask=visual_attn_mask)

class NextDiTCrossAttnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LuminaNextDiT2DModel(
            sample_size=DIT_INPUT_SIZE,
            patch_size=DIT_PATCH_SIZE,
            in_channels=DIT_IN_CHANNELS,
            hidden_size=DIT_HIDDEN_SIZE,
            num_layers=DIT_NUM_LAYERS,
            num_attention_heads=DIT_NUM_HEADS,
            num_kv_heads=DIT_NUM_HEADS,
            multiple_of=256,
            norm_eps=1e-5,
            learn_sigma=False,
            qk_norm=True,
            cross_attention_dim=DIT_LATENT_DIM,
        )

    def forward(self, x, timestep, z_latents):
        return self.model(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=z_latents,
            encoder_mask=torch.ones((z_latents.shape[0], z_latents.shape[1]), device=z_latents.device),
            image_rotary_emb=None,
            cross_attention_kwargs=dict(),
        ).sample

class InternVLAN1System1(nn.Module):
    def __init__(self, async_mode=True, model_base_path=None):
        super().__init__()
        self.async_mode = async_mode
        self.model_base_path = model_base_path
        
        self.traj_dit = NextDiTCrossAttnModule()
        self.action_encoder = nn.Linear(3, DIT_HIDDEN_SIZE, bias=True)
        self.action_decoder = nn.Linear(DIT_HIDDEN_SIZE, 3, bias=True)
        
        # Cached scheduler to improve inference efficiency
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.pos_encoder_fn = SinusoidalPositionalEncoding(DIT_HIDDEN_SIZE)
        # Precompute up to 1024 steps to avoid recompilations and graph breaks during generation
        pos_ids_max = torch.arange(1024).unsqueeze(0)
        self.register_buffer("_precomputed_pos_emb", self.pos_encoder_fn(pos_ids_max), persistent=False)
        
        if self.async_mode:
            self.rgb_model = build_depthanythingv2(model_base_path) 
            self.memory_encoder = MemoryEncoder(hidden_size=DIT_HIDDEN_SIZE)
            self.rgb_resampler = QFormer()

        self.register_buffer("_resnet_mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_resnet_std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, traj_latents, images_dp, noise=None):
        # Default parameters for export
        return self.generate_traj(
            traj_latents, 
            images_dp,
            noise=noise,
            num_inference_steps=10, 
            num_sample_trajs=1
        )

    @torch.inference_mode()
    def generate_traj(
        self,
        traj_latents, 
        images_dp,
        noise=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ):
        device = self.device
        dtype = traj_latents.dtype
        
        # 1. Condition feature extraction and Resampling
        if self.async_mode:
            images_dp = images_dp.permute(0, 1, 4, 2, 3)
            images_dp_norm = (images_dp - self._resnet_mean.to(dtype)) / self._resnet_std.to(dtype)
            
            images_dp_feat = (
                self.rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                .unflatten(dim=0, sizes=(1, -1))
            )
            memory_feat = self.memory_encoder(images_dp_feat.flatten(1, 2))
            memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
            memory_tokens = self.rgb_resampler(memory_feat)
            hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
        else:
            hidden_states = traj_latents
            
        # 2. CFG setup and Noise initialization
        hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
        hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)
        
        if noise is not None:
            latents = noise
        else:
            latents = torch.randn(
                (traj_latents.shape[0] * num_sample_trajs, predict_step_nums, 3),
                device=device, dtype=dtype
            )
        
        # 3. Retrieve precomputed pos emb for sequence efficiency (Graph Break strictly eliminated)
        pos_embeddings = self._precomputed_pos_emb[:, :predict_step_nums, :].to(device=device, dtype=dtype)
        
        # 4. Iterative Trajectory Generation Loop
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        
        for t in self.scheduler.timesteps:
            latent_features = self.action_encoder(latents)
            latent_features += pos_embeddings
            
            latent_model_input = torch.cat([latent_features, latent_features], dim=0)
            
            noise_pred = self.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device, torch.long),
                z_latents=hidden_states_input,
            )
            noise_pred = self.action_decoder(noise_pred)
            
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents

    @classmethod
    def from_pretrained_system1(cls, model_path, device="cpu", dtype=torch.float32):
        # Auto-detect base path for sibling modules
        model_base_path = os.path.dirname(os.path.dirname(model_path))
        model = cls(model_base_path=model_base_path).to(device=device, dtype=dtype)
        
        state_dict = load_file(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[6:] if k.startswith("model.") else k
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        return model
