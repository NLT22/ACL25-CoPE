"""
Encoder and model architectures for Probabilistic CLIP.

This module contains the core model architecture including encoders,
transformers, and the main ProbCLIPModel that integrates vision and text.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.clip.modeling_clip import (
    CLIPModel,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionTransformer,
    CLIPVisionModel,
    CLIPTextModel,
    CLIPTextTransformer,
)
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.models.clip.modeling_clip import BaseModelOutput, BaseModelOutputWithPooling

from .config import (
    ProbCLIPConfig, 
    ProbCLIPVisionConfig, 
    ProbCLIPTextConfig,
    ProbBaseModelOutput,
    ProbBaseModelOutputWithPooling
)
from .utils import CrossAttentionModulationBlock, UncertaintyPooler
from .loss import CoPELoss


class ProbCLIPEncoder(CLIPEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.n_unc_layers = config.n_unc_layers
        self.unc_layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(self.n_unc_layers)])
        self.modulation_block = CrossAttentionModulationBlock()
        
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        feat_modulate: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states or isinstance(feat_modulate, torch.Tensor) else None
        encoder_states_var = () if output_hidden_states or isinstance(feat_modulate, torch.Tensor) else None
        all_attentions = () if output_attentions else None
        all_attentions_var = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if idx == self.num_hidden_layers - self.n_unc_layers:
                # start of uncertainty branch
                hidden_states_var = hidden_states
            
            if output_hidden_states or isinstance(feat_modulate, torch.Tensor):
                encoder_states = encoder_states + (hidden_states,)
                if idx < self.num_hidden_layers - self.n_unc_layers:
                    # shared part of mean and std branch
                    encoder_states_var = encoder_states_var + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )
        
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if idx < self.num_hidden_layers - self.n_unc_layers:
                    # shared part of mean and std branch
                    all_attentions_var = all_attentions_var + (layer_outputs[1],)
        
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # modulation by text feature
        # encoder_states: ([B, L, D_i] * N_Layers), D_i = 1024, feat_modulate: [B, D_t], D_t = 768
        if feat_modulate is not None:
            modulated_encoder_states = ()
            for intermediate_state in encoder_states:
                modulated_encoder_states = modulated_encoder_states +\
                     (intermediate_state + self.modulation_block(intermediate_state, feat_modulate),)
        # mean pooling for modulated encoder states: (B, L, D_i) * N_Layers
            hidden_states = torch.mean(torch.stack(modulated_encoder_states), dim=0) # [B, L, D_i]

        if self.n_unc_layers == 0:
            hidden_states_var = hidden_states
        
        # uncertainty branch
        for idx, unc_layer in enumerate(self.unc_layers):

            if output_hidden_states:
                encoder_states_var = encoder_states_var + (hidden_states_var,)

            unc_layer_outputs = unc_layer(
                hidden_states_var,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states_var = unc_layer_outputs[0]

            if output_attentions:
                all_attentions_var = all_attentions_var + (unc_layer_outputs[1],)

        if output_hidden_states:
            encoder_states_var = encoder_states_var + (hidden_states_var,)

        return ProbBaseModelOutput(
            last_hidden_state=hidden_states,
            last_hidden_state_var=hidden_states_var,
            hidden_states=encoder_states,
            hidden_states_var=encoder_states_var,
            attentions=all_attentions,
            attentions_var=all_attentions_var
        )


class ProbCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: ProbCLIPVisionConfig):
        super().__init__(config)
        self.config = config
        self.n_unc_layers = config.n_unc_layers
        self.uncertainty_head = None
        self.encoder = ProbCLIPEncoder(config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        feat_modulate: Optional[torch.FloatTensor] = None,
    ) -> ProbBaseModelOutputWithPooling:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: ProbBaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            feat_modulate=feat_modulate,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return ProbBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            last_hidden_state_var=encoder_outputs.last_hidden_state_var,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            hidden_states_var=encoder_outputs.hidden_states_var,
            attentions=encoder_outputs.attentions,
            attentions_var=encoder_outputs.attentions_var,
        )


class ProbCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: ProbCLIPVisionConfig):
        super().__init__(config)
        self.config = config
        assert config.n_unc_layers >= 0, 'n_unc_layers must be greater than 0'
        assert config.n_unc_layers <= config.num_hidden_layers, 'n_unc_layers must be no greater than num_hidden_layers'
        self.n_unc_layers = config.n_unc_layers
        self.uncertainty_head = None
        self.vision_model = ProbCLIPVisionTransformer(config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        feat_modulate: Optional[torch.FloatTensor] = None,
    ) -> ProbBaseModelOutputWithPooling:

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            feat_modulate=feat_modulate,
        )


class ProbCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config: ProbCLIPTextConfig):
        super().__init__(config)
        self.config = config
        self.n_unc_layers = config.n_unc_layers
        self.encoder = ProbCLIPEncoder(config)
        
        # For attention mask, it differs between `flash_attention_2` and other attention implementations
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ProbBaseModelOutputWithPooling:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: ProbBaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        return ProbBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            last_hidden_state_var=encoder_outputs.last_hidden_state_var,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            hidden_states_var=encoder_outputs.hidden_states_var,
            attentions=encoder_outputs.attentions,
            attentions_var=encoder_outputs.attentions_var,
        )


class ProbCLIPTextModel(CLIPTextModel):
    def __init__(self, config: ProbCLIPTextConfig):
        super().__init__(config)
        self.config = config
        assert config.n_unc_layers >= 0, 'n_unc_layers must be greater than 0'
        assert config.n_unc_layers <= config.num_hidden_layers, 'n_unc_layers must be no greater than num_hidden_layers'
        self.n_unc_layers = config.n_unc_layers
        self.text_model = ProbCLIPTextTransformer(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ProbBaseModelOutputWithPooling:

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class ProbCLIPModel(CLIPModel):
    config_class = ProbCLIPConfig
    _no_split_modules = ["CLIPTextEmbeddings", "ProbCLIPEncoderLayer", "CLIPVisionEmbeddings"]

    def __init__(self, config: ProbCLIPConfig):
        # Call PretrainedModel.__init__ directly to bypass CLIPModel's type validation
        # since we use probabilistic config types that inherit from the base types
        super(CLIPModel, self).__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        text_model = ProbCLIPTextModel._from_config(text_config)
        self.text_model = text_model.text_model

        vision_model = ProbCLIPVisionModel._from_config(vision_config)
        self.vision_model = vision_model.vision_model

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # uncertainty poolers
        self.vision_uncertainty_pooler = UncertaintyPooler(
            self.vision_embed_dim, 
            self.projection_dim,
            sigma_ln_init=vision_config.sigma_ln_init,
            sigma_ln_init_bias=vision_config.sigma_ln_init_bias
        )
        self.text_uncertainty_pooler = UncertaintyPooler(
            self.text_embed_dim, 
            self.projection_dim,
            sigma_ln_init=text_config.sigma_ln_init,
            sigma_ln_init_bias=text_config.sigma_ln_init_bias
        )
        
        # PCME++ loss for probabilistic composed image retrieval
        self.loss = CoPELoss()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)

        """Initialize custom modules"""
        if isinstance(module, UncertaintyPooler):
            module._initialize_weights()


    @property
    def vision_std_ln(self):
        """Access to vision uncertainty pooler's LayerNorm for monitoring"""
        return self.vision_uncertainty_pooler.std_ln
    
    @property
    def text_std_ln(self):
        """Access to text uncertainty pooler's LayerNorm for monitoring"""
        return self.text_uncertainty_pooler.std_ln
    
    def get_image_features(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        feat_modulate: Optional[torch.FloatTensor] = None,
    ) -> dict[str, torch.FloatTensor]:
        """
        Extract image features with uncertainty estimation.
        
        Returns:
            dict with keys:
                - 'mean': image feature means
                - 'var': log(variance) of image features
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            feat_modulate=feat_modulate
        )

        pooled_output = vision_outputs.pooler_output
        output_var = vision_outputs.last_hidden_state_var
        image_features_mean = self.visual_projection(pooled_output)
        image_features_var = self.vision_uncertainty_pooler(output_var)

        return {
            'mean': image_features_mean,
            'var': image_features_var,  # This is log(var) from uncertainty pooler
        }

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> dict[str, torch.FloatTensor]:
        """
        Extract text features with uncertainty estimation.
        
        Returns:
            dict with keys:
                - 'mean': text feature means
                - 'var': log(variance) of text features
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs: ProbBaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs.pooler_output
        output_var = text_outputs.last_hidden_state_var
        text_features_mean = self.text_projection(pooled_output)
        text_features_var = self.text_uncertainty_pooler(output_var)

        return {
            'mean': text_features_mean,
            'var': text_features_var,  # This is log(var) from uncertainty pooler
        }
