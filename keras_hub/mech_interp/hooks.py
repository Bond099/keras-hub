import keras
import numpy as np
from keras_hub.src.models.backbone import Backbone as BaseBackbone
from keras_hub.src.models.causal_lm import CausalLM as BaseCausalLM
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.mech_interp.gemma_hooked import HookedGemmaDecoderBlock, HookedCachedGemmaAttention
from keras_hub.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock
from keras_hub.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_hub.mech_interp.utils import get_act_name

HOOKED_DECODER_REGISTRY = {
    'gemma': {
        'orig_cls': GemmaDecoderBlock,
        'hooked_cls': HookedGemmaDecoderBlock,
        'attr_name': 'transformer_layers',
        'is_list': True,
        'attention_orig_cls': CachedGemmaAttention,
        'attention_hooked_cls': HookedCachedGemmaAttention,
    },
    # Example for Mistral (add hooked classes similarly)
    # 'mistral': {
    #     'orig_cls': MistralTransformerDecoder,
    #     'hooked_cls': HookedMistralTransformerDecoder,
    #     'attr_name': 'transformer_layers',
    #     'is_list': True,
    #     'attention_orig_cls': CachedMistralAttention,
    #     'attention_hooked_cls': HookedCachedMistralAttention,
    # },
}

class ActivationCache(dict):
    def remove_batch_dim(self):
        for k, v in self.items():
            if isinstance(v, (keras.KerasTensor, np.ndarray)) and len(v.shape) > 0 and v.shape[0] == 1:
                self[k] = np.squeeze(v, axis=0) if isinstance(v, np.ndarray) else ops.squeeze(v, axis=0)

class HookedBackbone(BaseBackbone):
    def __init__(self, backbone, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.hooks = {}  # Global hooks (e.g., 'hook_embed')

    @classmethod
    def from_preset(cls, preset, load_weights=True, **kwargs):
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        backbone = loader.load_backbone(backbone_cls, load_weights, **kwargs)
        hooked_backbone = cls(backbone)
        hooked_backbone._replace_with_hooked_decoders(preset)
        return hooked_backbone

    def _replace_with_hooked_decoders(self, preset):
        family = next((key for key in HOOKED_DECODER_REGISTRY if key in preset.lower()), None)
        if family is None:
            print(f"Warning: No hooked decoder support for {preset}; using coarse-grained hooks.")
            return
        config = HOOKED_DECODER_REGISTRY[family]
        orig_cls = config['orig_cls']
        hooked_cls = config['hooked_cls']
        attr_name = config['attr_name']
        is_list = config['is_list']
        attn_orig = config.get('attention_orig_cls')
        attn_hooked = config.get('attention_hooked_cls')

        layers = getattr(self.backbone, attr_name)
        if not is_list or not isinstance(layers, list):
            raise ValueError(f"Expected {attr_name} to be a list for {preset}.")
        
        for i, layer in enumerate(layers):
            if isinstance(layer, orig_cls):
                hooked_layer = hooked_cls(**layer.get_config())
                hooked_layer.build(layer.input_shape if hasattr(layer, 'input_shape') else None)
                hooked_layer.set_weights(layer.get_weights())
                layers[i] = hooked_layer
                if attn_orig and attn_hooked and isinstance(hooked_layer.attention, attn_orig):
                    attn = hooked_layer.attention
                    hooked_attn = attn_hooked(**attn.get_config())
                    hooked_attn.build(attn.input_shape if hasattr(attn, 'input_shape') else None)
                    hooked_attn.set_weights(attn.get_weights())
                    hooked_layer.attention = hooked_attn

        # Hook final layer_norm
        if hasattr(self.backbone, 'layer_norm'):
            self.backbone.layer_norm = self._hook_simple_layer(self.backbone.layer_norm, 'normalized')

    def _hook_simple_layer(self, layer, hook_name):
        class HookedSimpleLayer(keras.layers.Layer):
            def __init__(self, layer, hook_name, **kwargs):
                super().__init__(**kwargs)
                self.layer = layer
                self.hook_name = hook_name
                self.hooks = {}

            def add_hook(self, local_name, hook_fn):
                self.hooks.setdefault(local_name, []).append(hook_fn)

            def _apply_hooks(self, act, local_name):
                full_name = self.hook_name if local_name == '' else f"{self.hook_name}_{local_name}"
                if full_name in self.hooks:
                    for fn in self.hooks[full_name]:
                        mod_act = fn(act)
                        if mod_act is not None:
                            act = mod_act
                return act

            def call(self, inputs, **kwargs):
                act = self._apply_hooks(inputs, 'input')
                act = self.layer(act, **kwargs)
                return self._apply_hooks(act, '')

        hooked = HookedSimpleLayer(layer, hook_name, name=layer.name, dtype=self.dtype_policy)
        return hooked

    def add_hook(self, hook_name, hook_fn):
        if hook_name.startswith('blocks.'):
            parts = hook_name.split('.')
            block_idx = int(parts[1])
            local_name = '.'.join(parts[2:])
            block = self.backbone.transformer_layers[block_idx]
            if local_name.startswith('hook_attn_'):
                block.attention.add_hook(local_name.replace('hook_', ''), hook_fn)
            else:
                block.add_hook(local_name.replace('hook_', ''), hook_fn)
        elif hook_name == 'hook_embed':
            # Route to embedding hook
            def embed_hook(act):
                mod_act = hook_fn(act)
                return mod_act if mod_act is not None else act
            # Assume embed is hooked by overriding, or add temp
            self.hooks['embed'] = embed_hook
        elif hook_name == 'hook_normalized':
            self.backbone.layer_norm.add_hook('', hook_fn)
        else:
            self.hooks.setdefault(hook_name, []).append(hook_fn)

    def reset_hooks(self):
        self.hooks = {}
        for block in self.backbone.transformer_layers:
            block.hooks = {}
            block.attention.hooks = {}
        self.backbone.layer_norm.hooks = {}

    def call(self, inputs, training=False, cache=None, cache_update_index=0):
        token_ids = inputs['token_ids'] if isinstance(inputs, dict) else inputs
        padding_mask = inputs.get('padding_mask') if isinstance(inputs, dict) else None
        
        x = self.backbone.token_embedding(token_ids)
        x = x * ops.cast(ops.sqrt(self.backbone.hidden_dim), x.dtype)
        x = self._apply_hooks(x, 'embed')

        caches = []
        for i, block in enumerate(self.backbone.transformer_layers):
            if cache is not None:
                current_cache = cache[i] if isinstance(cache, list) else cache[:, i, ...]
                x, new_cache = block(x, padding_mask=padding_mask, cache=current_cache, cache_update_index=cache_update_index)
                caches.append(new_cache)
            else:
                x = block(x, padding_mask=padding_mask)
        
        if cache is not None:
            cache = caches if isinstance(cache, list) else ops.stack(caches, axis=0)

        x = self.backbone.layer_norm(x)
        x = self._apply_hooks(x, 'normalized')

        return x

    def _apply_hooks(self, act, hook_name):
        if hook_name in self.hooks:
            for fn in self.hooks[hook_name]:
                mod_act = fn(act)
                if mod_act is not None:
                    act = mod_act
        return act

    def run_with_hooks(self, inputs, fwd_hooks=None, training=False):
        fwd_hooks = fwd_hooks or []
        old_hooks = self.hooks.copy()
        for hook_name, hook_fn in fwd_hooks:
            self.add_hook(hook_name, hook_fn)
        outputs = self.call(inputs, training=training)
        self.hooks = old_hooks
        return outputs

    def run_with_cache(self, inputs, return_type="outputs", names_filter=None, training=False):
        cache = ActivationCache()
        def collect_fn(act, hook_name):
            if names_filter is None or names_filter(hook_name):
                cache[hook_name] = act
            return act

        all_hooks = []
        for hook_type in ['resid_pre', 'ln0', 'attn_q', 'attn_k', 'attn_v', 'attn_pattern', 'attn_z', 'attn_out', 'ln1', 'resid_mid', 'ln2', 'mlp_out', 'ln3', 'resid_post']:
            for i in range(self.backbone.num_layers):
                all_hooks.append((get_act_name(hook_type, i), lambda act: collect_fn(act, get_act_name(hook_type, i))))
        all_hooks.append(('hook_embed', lambda act: collect_fn(act, 'hook_embed')))
        all_hooks.append(('hook_normalized', lambda act: collect_fn(act, 'hook_normalized')))

        outputs = self.run_with_hooks(inputs, fwd_hooks=all_hooks, training=training)
        if return_type == "outputs":
            return outputs
        elif return_type == "both":
            return outputs, cache
        elif return_type == "cache":
            return cache
        raise ValueError(f"Invalid return_type: {return_type}")

class HookedCausalLM(BaseCausalLM):
    @classmethod
    def from_preset(cls, preset, load_weights=True, **kwargs):
        hooked_backbone = HookedBackbone.from_preset(preset, load_weights=load_weights, **kwargs)
        base_lm = BaseCausalLM.from_preset(preset)
        return cls(backbone=hooked_backbone, preprocessor=base_lm.preprocessor)

    def add_hook(self, *args, **kwargs):
        self.backbone.add_hook(*args, **kwargs)

    def reset_hooks(self):
        self.backbone.reset_hooks()

    def run_with_hooks(self, *args, **kwargs):
        return self.backbone.run_with_hooks(*args, **kwargs)

    def run_with_cache(self, *args, **kwargs):
        return self.backbone.run_with_cache(*args, **kwargs)

    def generate(self, inputs, max_length=None, stop_token_ids="auto", strip_prompt=False, return_cache=False):
        if return_cache:
            self.temp_cache = ActivationCache()
            def collect_fn(act, hook_name):
                if hook_name in self.temp_cache:
                    self.temp_cache[hook_name].append(act)
                else:
                    self.temp_cache[hook_name] = [act]
                return act

            all_hooks = []
            for hook_type in ['resid_pre', 'ln0', 'attn_q', 'attn_k', 'attn_v', 'attn_pattern', 'attn_z', 'attn_out', 'ln1', 'resid_mid', 'ln2', 'mlp_out', 'ln3', 'resid_post']:
                for i in range(self.backbone.backbone.num_layers):
                    all_hooks.append((get_act_name(hook_type, i), lambda act: collect_fn(act, get_act_name(hook_type, i))))
            all_hooks.append(('hook_embed', lambda act: collect_fn(act, 'hook_embed')))
            all_hooks.append(('hook_normalized', lambda act: collect_fn(act, 'hook_normalized')))

            for hook_name, fn in all_hooks:
                self.add_hook(hook_name, fn)

        outputs = super().generate(inputs, max_length=max_length, stop_token_ids=stop_token_ids, strip_prompt=strip_prompt)

        if return_cache:
            self.reset_hooks()
            cache = self.temp_cache
            del self.temp_cache
            return outputs, cache

        return outputs