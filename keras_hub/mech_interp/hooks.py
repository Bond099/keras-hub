import keras
from keras import ops
import numpy as np
import importlib
from keras_hub.src.models.backbone import Backbone as BaseBackbone
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.models.causal_lm import CausalLM as BaseCausalLM
from keras_hub.mech_interp.utils import get_act_name

HOOKED_DECODER_REGISTRY = {
    'gemma': {
        'module_name': 'keras_hub.src.models.gemma.gemma_decoder_block',
        'class_name': 'GemmaDecoderBlock',
        'hooked_module_name': 'keras_hub.mech_interp.gemma_hooked',
        'hooked_class_name': 'HookedGemmaDecoderBlock',
        'attr_name': 'transformer_layers',
        'is_list': True,
        'hook_function': 'hook_layer',
        'default_hook_types': [
            'resid_pre', 'ln0', 'attn_q', 'attn_rot_q', 'attn_k', 'attn_rot_k', 'attn_v',
            'attn_scores', 'attn_pattern', 'attn_z', 'attn_out', 'resid_mid', 'ln1',
            'ln2', 'mlp_gated1', 'mlp_gated2', 'mlp_out', 'ln3', 'resid_post'
        ],
        # Add fields for future models, e.g., 'attention_class_name': 'GemmaAttention',
    },
    # Add other models...
}

class HookMixin:
    def __init__(self):
        self.hooks = {}

    def add_hook(self, hook_name, hook_fn):
        self.hooks.setdefault(hook_name, []).append(hook_fn)

    def _apply_hooks(self, act, hook_name):
        if hook_name not in self.hooks:
            return act
        for fn in self.hooks[hook_name]:
            mod_act = fn(act)
            if mod_act is not None:
                act = mod_act
        return act

    def reset_hooks(self):
        self.hooks = {}

class ActivationCache(dict):
    def remove_batch_dim(self):
        for k, v in self.items():
            if ops.shape(v)[0] == 1:
                self[k] = ops.squeeze(v, axis=0)
        return self

class BaseHookedModel(HookMixin):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def run_with_hooks(self, inputs, fwd_hooks=None, training=False):
        fwd_hooks = fwd_hooks or []
        old_hooks = self.hooks.copy()
        for hook_name, hook_fn in fwd_hooks:
            self.add_hook(hook_name, hook_fn)
        outputs = self(inputs, training=training)
        self.hooks = old_hooks
        return outputs

    def run_with_cache(self, inputs, return_type="both", names_filter=None, training=False):
        cache = ActivationCache()
        all_hooks = []
        hook_types = self.get_default_hook_types()
        for hook_type in hook_types:
            for i in range(self.get_num_layers()):
                full_name = get_act_name(hook_type, i)
                if names_filter is None or names_filter(full_name):
                    all_hooks.append((full_name, lambda act, fn=full_name: (cache.update({fn: act}), act)[1]))
        if names_filter is None or names_filter('hook_embed'):
            all_hooks.append(('hook_embed', lambda act: (cache.update({'hook_embed': act}), act)[1]))
        if names_filter is None or names_filter('hook_normalized'):
            all_hooks.append(('hook_normalized', lambda act: (cache.update({'hook_normalized': act}), act)[1]))
        outputs = self.run_with_hooks(inputs, fwd_hooks=all_hooks, training=training)
        if return_type == "outputs":
            return outputs
        elif return_type == "both":
            return outputs, cache
        elif return_type == "cache":
            return cache
        raise ValueError(f"Invalid return_type: {return_type}")

    def get_default_hook_types(self):
        raise NotImplementedError("Subclasses must implement get_default_hook_types")

    def get_num_layers(self):
        raise NotImplementedError("Subclasses must implement get_num_layers")

class HookedBackbone(BaseHookedModel, BaseBackbone):
    def __init__(self, backbone, **kwargs):
        BaseHookedModel.__init__(self, backbone)
        self._wrap_layers()
        token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask_input = keras.Input(shape=(None,), dtype="int32", name="padding_mask")
        inputs = {"token_ids": token_id_input, "padding_mask": padding_mask_input}
        outputs = self.call(inputs)
        BaseBackbone.__init__(self, inputs, outputs, **kwargs)

    def _wrap_layers(self):
        self.backbone.layer_norm = self._hook_simple_layer(self.backbone.layer_norm, 'normalized')

    @classmethod
    def from_preset(cls, preset, load_weights=True, **kwargs):
        return load_hooked_from_preset(preset, cls, load_weights=load_weights, is_causal_lm=False, **kwargs)

    @staticmethod
    def _hook_simple_layer(layer, hook_name):
        class HookedSimpleLayer(keras.layers.Layer, HookMixin):
            def __init__(self, *args, **kwargs):
                keras.layers.Layer.__init__(self, *args, **kwargs)
                HookMixin.__init__(self)

            def call(self, inputs, **kwargs):
                act = self._apply_hooks(inputs, hook_name + '_input')
                act = layer(act, **kwargs)
                return self._apply_hooks(act, hook_name)
        return HookedSimpleLayer(name=layer.name)

    def add_hook(self, hook_name, hook_fn):
        if hook_name.startswith('blocks.'):
            parts = hook_name.split('.')
            block_idx = int(parts[1])
            local_name = '.'.join(parts[2:]).replace('hook_', '')
            block = self.backbone.transformer_layers[block_idx]
            if local_name.startswith('attn_'):
                block.attention.add_hook(local_name, hook_fn)
            else:
                block.add_hook(local_name, hook_fn)
        elif hook_name == 'hook_embed':
            super().add_hook('embed', hook_fn)
        elif hook_name == 'hook_normalized':
            self.backbone.layer_norm.add_hook('normalized', hook_fn)
        else:
            super().add_hook(hook_name, hook_fn)

    def reset_hooks(self):
        super().reset_hooks()
        for block in self.backbone.transformer_layers:
            block.reset_hooks()
            block.attention.reset_hooks()
        self.backbone.layer_norm.reset_hooks()

    def call(self, inputs, cache=None, cache_update_index=0, training=False):
        token_ids = inputs['token_ids']
        padding_mask = inputs['padding_mask']
        x = self.backbone.token_embedding(token_ids)
        x = x * ops.cast(ops.sqrt(self.backbone.hidden_dim), dtype=x.dtype)
        x = self._apply_hooks(x, 'embed')
        updated_caches = []
        for i, block in enumerate(self.backbone.transformer_layers):
            if cache is not None:
                layer_cache = cache[i]
                x, new_layer_cache = block(x, padding_mask=padding_mask, cache=layer_cache, cache_update_index=cache_update_index)
                updated_caches.append(new_layer_cache)
            else:
                x = block(x, padding_mask=padding_mask)
        x = self.backbone.layer_norm(x)
        x = self._apply_hooks(x, 'normalized')
        if cache is not None:
            return x, updated_caches
        return x

    def get_default_hook_types(self):
        family = self._get_family_from_preset()
        return HOOKED_DECODER_REGISTRY.get(family, {}).get('default_hook_types', [])

    def get_num_layers(self):
        return len(self.backbone.transformer_layers)

    def _get_family_from_preset(self):
        # Assume preset is stored or derivable; for simplicity, check backbone name
        for key in HOOKED_DECODER_REGISTRY:
            if key in self.backbone.name.lower():
                return key
        return None

class HookedCausalLM(BaseHookedModel, BaseCausalLM):
    def __init__(self, backbone, preprocessor=None, **kwargs):
        BaseHookedModel.__init__(self, backbone)
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.backbone.token_embedding(hidden_states, reverse=True)
        BaseCausalLM.__init__(self, inputs=inputs, outputs=outputs, **kwargs)
        self.preprocessor = preprocessor

    @classmethod
    def from_preset(cls, preset, load_weights=True, **kwargs):
        return load_hooked_from_preset(preset, cls, load_weights=load_weights, is_causal_lm=True, **kwargs)

    def call(self, inputs, cache=None, cache_update_index=0, training=False):
        hidden_states = self.backbone(inputs, cache=cache, cache_update_index=cache_update_index, training=training)
        if cache is not None:
            hidden_states, cache = hidden_states
        outputs = self.backbone.backbone.token_embedding(hidden_states, reverse=True)
        if cache is not None:
            return outputs, cache
        return outputs

    def generate_preprocess(self, x, sequence_length=None):
        if self.preprocessor is None:
            raise AttributeError("Preprocessor is None; cannot preprocess raw input. Provide preprocessed dict or set a preprocessor.")
        inputs = self.preprocessor.generate_preprocess(x, sequence_length=sequence_length)
        if not isinstance(x, (list, tuple)):
            inputs['token_ids'] = ops.expand_dims(inputs['token_ids'], axis=0)
            inputs['padding_mask'] = ops.expand_dims(ops.cast(inputs['padding_mask'], 'int32'), axis=0)
        return inputs

    def add_hook(self, *args, **kwargs):
        self.backbone.add_hook(*args, **kwargs)

    def reset_hooks(self):
        self.backbone.reset_hooks()

    def run_with_hooks(self, *args, **kwargs):
        return self.backbone.run_with_hooks(*args, **kwargs)

    def run_with_cache(self, *args, **kwargs):
        return self.backbone.run_with_cache(*args, **kwargs)

    def generate(self, inputs, max_length=None, stop_token_ids="auto", strip_prompt=False, return_cache=False):
        if return_cache and not isinstance(inputs, (str, dict)):
            raise ValueError("return_cache supported only for single input; use single str or dict for batched=1.")
        was_single = isinstance(inputs, (str, dict))
        if isinstance(inputs, str):
            inputs = [self.generate_preprocess(inputs, sequence_length=max_length)]
        elif isinstance(inputs, dict):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise ValueError("Inputs must be str, dict, or list of str/dict.")
        inputs = [self.generate_preprocess(x, sequence_length=max_length) if isinstance(x, str) else x for x in inputs]
        if stop_token_ids == "auto":
            if self.preprocessor is None:
                raise ValueError("'auto' stop_token_ids requires preprocessor; provide explicit IDs.")
            stop_token_ids = [self.preprocessor.tokenizer.end_token_id]
        outputs = []
        for input_dict in inputs:
            token_ids = input_dict['token_ids']
            padding_mask = input_dict['padding_mask']
            prompt_len = ops.sum(ops.cast(padding_mask[0], 'int32'))  # Batch=1
            current_len = prompt_len
            seq_len = ops.shape(token_ids)[1]
            if seq_len > max_length:
                input_dict['token_ids'] = token_ids[:, :max_length]
                input_dict['padding_mask'] = padding_mask[:, :max_length]
                token_ids = input_dict['token_ids']
                padding_mask = input_dict['padding_mask']
                seq_len = max_length
            # Initialize cache
            num_layers = len(self.backbone.backbone.transformer_layers)
            num_kv_heads = self.backbone.backbone.num_key_value_heads
            head_dim = self.backbone.backbone.head_dim
            batch = ops.shape(token_ids)[0]
            dtype = self.dtype_policy.compute_dtype
            cache = [ops.zeros((batch, 2, seq_len, num_kv_heads, head_dim), dtype=dtype) for _ in range(num_layers)]
            # Initial forward on full padded prompt to build cache and get first next_token logits
            prompt_inputs = input_dict
            logits, cache = self(prompt_inputs, cache=cache, cache_update_index=0)
            next_token = ops.argmax(logits[0, prompt_len - 1])
            current_len = prompt_len
            while current_len < max_length:
                if next_token in stop_token_ids:
                    break
                token_ids = ops.slice_update(token_ids, [0, current_len], ops.reshape(next_token, (1, 1)))
                padding_mask = ops.slice_update(padding_mask, [0, current_len], [[1]])
                # Compute next logits from the last added token
                next_inputs = {'token_ids': token_ids[:, current_len:current_len + 1], 'padding_mask': padding_mask[:, current_len:current_len + 1]}
                logits, cache = self(next_inputs, cache=cache, cache_update_index=current_len)
                next_token = ops.argmax(logits[0, 0])
                current_len += 1
            output_tokens = token_ids[0, :current_len]
            full_tokens_for_cache = output_tokens
            if strip_prompt:
                output_tokens = output_tokens[prompt_len:]
            if self.preprocessor:
                postprocess_input = {"token_ids": ops.expand_dims(output_tokens, 0), "padding_mask": ops.ones((1, ops.shape(output_tokens)[0]), "int32")}
                output = self.preprocessor.generate_postprocess(postprocess_input)
                output = output[0]
            else:
                output = output_tokens
            outputs.append(output)
        if return_cache:
            cache_inputs = {"token_ids": ops.expand_dims(full_tokens_for_cache, 0), "padding_mask": ops.ones((1, ops.shape(full_tokens_for_cache)[0]), "int32")}
            _, cache = self.run_with_cache(cache_inputs, return_type="both")
            cache.remove_batch_dim()
            return outputs[0] if was_single else outputs, cache
        return outputs[0] if was_single else outputs

    def get_default_hook_types(self):
        return self.backbone.get_default_hook_types()

    def get_num_layers(self):
        return self.backbone.get_num_layers()

def load_hooked_from_preset(preset, hooked_cls, load_weights=True, is_causal_lm=False, **kwargs):
    loader = get_preset_loader(preset)
    backbone_cls = loader.check_backbone_class()
    if is_causal_lm:
        base_lm = BaseCausalLM.from_preset(preset, load_weights=load_weights, **kwargs)
        backbone = base_lm.backbone
        preprocessor = base_lm.preprocessor
    else:
        backbone = loader.load_backbone(backbone_cls, load_weights, **kwargs)
        preprocessor = None
    family = next((key for key in HOOKED_DECODER_REGISTRY if key in preset.lower()), None)
    if family:
        config = HOOKED_DECODER_REGISTRY[family]
        orig_module = importlib.import_module(config['module_name'])
        orig_cls = getattr(orig_module, config['class_name'])
        hooked_module = importlib.import_module(config['hooked_module_name'])
        hook_fn = getattr(hooked_module, config['hook_function'])
        layers = getattr(backbone, config['attr_name'])
        for i, layer in enumerate(layers):
            if isinstance(layer, orig_cls):
                hooked_layer = hook_fn(layer, backbone)
                layers[i] = hooked_layer
    hooked_backbone = HookedBackbone(backbone)
    if is_causal_lm:
        return hooked_cls(hooked_backbone, preprocessor)
    return hooked_cls(hooked_backbone)