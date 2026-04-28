import json
import os

import torch
import torch.nn as nn

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

TRAJ_TOKEN_INDEX = 151667
DIT_LATENT_SIZE = 768

class InternVLAN1System2(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        # Projector for latent compression to DiT dimension
        self.model.cond_projector = nn.Sequential(
            nn.Linear(config.hidden_size, DIT_LATENT_SIZE),
            nn.GELU(approximate='tanh'),
            nn.Linear(DIT_LATENT_SIZE, DIT_LATENT_SIZE)
        )
        
        self.n_query = getattr(config, 'n_query', 4)
        self.model.latent_queries = nn.Parameter(
            torch.randn(1, self.n_query, config.hidden_size)
        )
        self._last_projected_latents = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, **kwargs):
        input_ids = kwargs.pop('input_ids', None)
        pixel_values = kwargs.pop('pixel_values', None)
        image_grid_thw = kwargs.pop('image_grid_thw', None)
        attention_mask = kwargs.pop('attention_mask', None)
        position_ids = kwargs.pop('position_ids', None)
        inputs_embeds = kwargs.pop('inputs_embeds', None)

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if inputs_embeds.shape[1] < input_ids.shape[1]:
                diff = input_ids.shape[1] - inputs_embeds.shape[1]
                inputs_embeds = torch.cat([inputs_embeds, self.model.embed_tokens(input_ids[:, -diff:])], dim=1)

            # ViT image embedding (masked_scatter: graph-break-free)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                mask = (input_ids == self.config.image_token_id)
                mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        target_len = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]

        # Extend attention_mask if input_ids was extended (e.g., traj tokens appended by caller)
        # Only triggers during prefill (mask shorter than ids), never during decode (mask longer)
        if attention_mask is not None and input_ids is not None and attention_mask.shape[1] < input_ids.shape[1]:
            pad_len = input_ids.shape[1] - attention_mask.shape[1]
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, pad_len), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)

        # Inject learned trajectory queries at the end of the sequence
        if target_len >= self.n_query and input_ids is not None:
            is_traj_mask = (input_ids == TRAJ_TOKEN_INDEX).any(dim=1).view(-1, 1, 1).to(inputs_embeds.dtype)
            latent_queries = self.model.latent_queries.repeat(batch_size, 1, 1).to(inputs_embeds)

            base_embeds = inputs_embeds[:, :-self.n_query, :]
            tail_embeds = inputs_embeds[:, -self.n_query:, :]
            
            new_tail = tail_embeds * (1.0 - is_traj_mask) + latent_queries * is_traj_mask
            inputs_embeds = torch.cat([base_embeds, new_tail], dim=1)

        # Pass input_ids and inputs_embeds to the parent simultaneously:
        # - Since inputs_embeds is provided, the parent skips embedding lookup.
        # - With input_ids provided, the parent correctly calculates position_ids (including rope_deltas + cache_position).
        # - The parent always passes input_ids=None to self.model(), so there is no conflict.
        kwargs.update({
            'input_ids': input_ids,
            'inputs_embeds': inputs_embeds,
            'position_ids': None,
            'pixel_values': None,
            'image_grid_thw': image_grid_thw,
            'attention_mask': attention_mask,
            'output_hidden_states': True,
            'return_dict': True
        })

        outputs = super().forward(**kwargs)
        
        # Extract and project hidden states for System 1
        # Do not extract during decode steps (seq_len=1) — preserves the latents from the prefill phase.
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
            if last_hidden.shape[1] >= self.n_query:
                self._last_projected_latents = self.model.cond_projector(last_hidden[:, -self.n_query:, :]).detach()

        return outputs

    def get_last_latents(self):
        return self._last_projected_latents

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        traj_tokens = torch.full((input_ids.shape[0], self.n_query), TRAJ_TOKEN_INDEX,
                                 dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, traj_tokens], dim=1)
        self.forward(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        return self.get_last_latents()

    @classmethod
    def from_pretrained_system2(cls, model_path, **kwargs):
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict['model_type'] = 'qwen2_5_vl'
        config = Qwen2_5_VLConfig.from_dict(config_dict)
        config.do_sample = True  # Enable sampling to align with temperature settings
        return cls.from_pretrained(model_path, config=config, **kwargs)
