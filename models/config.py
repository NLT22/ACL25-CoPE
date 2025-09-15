"""
Configuration classes for Probabilistic CLIP models.

This module contains all configuration classes and output dataclasses
used by the probabilistic CLIP architecture.
"""

import torch
from dataclasses import dataclass
from typing import Optional
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPVisionConfig, CLIPTextConfig
from transformers.models.clip.modeling_clip import BaseModelOutput, BaseModelOutputWithPooling
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ProbCLIPVisionConfig(CLIPVisionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_unc_layers = kwargs.get('n_unc_layers', 1)
        self.sigma_ln_init = kwargs.get('sigma_ln_init', 0.01)
        self.sigma_ln_init_bias = kwargs.get('sigma_ln_init_bias', 0)


class ProbCLIPTextConfig(CLIPTextConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_unc_layers = kwargs.get('n_unc_layers', 1)
        self.sigma_ln_init = kwargs.get('sigma_ln_init', 0.01)
        self.sigma_ln_init_bias = kwargs.get('sigma_ln_init_bias', 0)


class ProbCLIPConfig(CLIPConfig):
    """
    Probabilistic CLIP configuration that uses ProbCLIPTextConfig and ProbCLIPVisionConfig
    for text_config and vision_config respectively, while keeping all other behaviors unchanged.
    """
    model_type = "prob_clip"
    sub_configs = {"text_config": ProbCLIPTextConfig, "vision_config": ProbCLIPVisionConfig}

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super(CLIPConfig, self).__init__(**kwargs)  # Call PretrainedConfig.__init__ directly

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = ProbCLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `ProbCLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = ProbCLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `ProbCLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `ProbCLIPTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `ProbCLIPVisionConfig` with default values.")

        # Use probabilistic configs instead of regular CLIP configs
        self.text_config = ProbCLIPTextConfig(**text_config)
        self.vision_config = ProbCLIPVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: ProbCLIPTextConfig, vision_config: ProbCLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`ProbCLIPConfig`] (or a derived class) from probabilistic clip text model configuration and 
        probabilistic clip vision model configuration.

        Returns:
            [`ProbCLIPConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


@dataclass
class ProbCLIPOutput(BaseModelOutput):
    mean: torch.FloatTensor | None = None
    var: torch.FloatTensor | None = None


@dataclass
class ProbBaseModelOutput(BaseModelOutput):
    last_hidden_state_var: Optional[torch.FloatTensor] = None
    hidden_states_var: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions_var: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class ProbBaseModelOutputWithPooling(BaseModelOutputWithPooling):
    last_hidden_state_var: torch.FloatTensor | None = None
    hidden_states_var: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions_var: Optional[tuple[torch.FloatTensor, ...]] = None
