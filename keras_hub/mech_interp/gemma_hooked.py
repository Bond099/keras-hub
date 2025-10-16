import keras
from keras import ops
from keras_hub.src.layers.modeling.transformer_layer_utils import compute_causal_mask, merge_padding_and_attention_mask
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_hub.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock

class HookedCachedGemmaAttention(CachedGemmaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = {}

    def add_hook(self, hook_name, hook_fn):
        self.hooks.setdefault(hook_name, []).append(hook_fn)

    def _apply_hooks(self, act, hook_name):
        if hook_name in self.hooks:
            for fn in self.hooks[hook_name]:
                mod_act = fn(act)
                if mod_act is not None:
                    act = mod_act
        return act

    def call(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=False,
    ):
        query = self.query_dense(x)
        query = self._apply_hooks(query, 'attn_q')
        query = self._apply_rope(query, cache_update_index)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.key_dense(x)
            key_update = self._apply_hooks(key_update, 'attn_k')
            key_update = self._apply_rope(key_update, cache_update_index)
            value_update = self.value_dense(x)
            value_update = self._apply_hooks(value_update, 'attn_v')
            key = ops.slice_update(key_cache, [0, cache_update_index, 0, 0], key_update)
            value = ops.slice_update(value_cache, [0, cache_update_index, 0, 0], value_update)
            cache = ops.stack((key, value), axis=1)
        else:
            key = self.key_dense(x)
            key = self._apply_hooks(key, 'attn_k')
            key = self._apply_rope(key, cache_update_index)
            value = self.value_dense(x)
            value = self._apply_hooks(value, 'attn_v')

        attention_vec = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            training=training,
            cache_update_index=cache_update_index,
        )
        attention_vec = self._apply_hooks(attention_vec, 'attn_z')  # z is pre-reshape

        attention_output = self.output_dense(attention_vec)

        # Wipe attn vec if no attended tokens
        no_attended_tokens = ops.all(ops.equal(attention_mask, 0), axis=-1, keepdims=True)[..., None]
        attention_output = ops.where(no_attended_tokens, ops.zeros_like(attention_output), attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

class HookedGemmaDecoderBlock(GemmaDecoderBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = {}
        self.attention = HookedCachedGemmaAttention(
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            logit_soft_cap=self.logit_soft_cap,
            use_sliding_window_attention=self.use_sliding_window_attention,
            sliding_window_size=self.sliding_window_size,
            query_head_dim_normalize=self.query_head_dim_normalize,
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="attention",
        )

    def add_hook(self, hook_name, hook_fn):
        self.hooks.setdefault(hook_name, []).append(hook_fn)

    def _apply_hooks(self, act, hook_name):
        if hook_name in self.hooks:
            for fn in self.hooks[hook_name]:
                mod_act = fn(act)
                if mod_act is not None:
                    act = mod_act
        return act

    def call(
        self,
        x,
        padding_mask=None,
        cache=None,
        cache_update_index=0,
    ):
        x = self._apply_hooks(x, 'resid_pre')

        normalized_x = self.pre_attention_norm(x)
        normalized_x = self._apply_hooks(normalized_x, 'ln0')

        attention_mask = self._compute_attention_mask(
            normalized_x, padding_mask, cache, cache_update_index
        )
        if cache is not None:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
            )
        else:
            attention = self.attention(
                normalized_x,
                attention_mask=attention_mask,
            )

        attention = self._apply_hooks(attention, 'attn_out')

        if self.use_post_attention_norm:
            attention = self.post_attention_norm(attention)
            attention = self._apply_hooks(attention, 'ln1')

        if self.dropout > 0:
            attention = self.attention_dropout(attention)

        attention_x = x + attention
        attention_x = self._apply_hooks(attention_x, 'resid_mid')

        normalized_x = self.pre_ffw_norm(attention_x)
        normalized_x = self._apply_hooks(normalized_x, 'ln2')

        x1 = self.gating_ffw(normalized_x)
        x2 = self.gating_ffw_2(normalized_x)
        x = keras.activations.gelu(x1, approximate=True) * x2
        x = self.ffw_linear(x)
        x = self._apply_hooks(x, 'mlp_out')

        if self.use_post_ffw_norm:
            x = self.post_ffw_norm(x)
            x = self._apply_hooks(x, 'ln3')

        x = x + attention_x
        x = self._apply_hooks(x, 'resid_post')

        if cache is not None:
            return x, new_cache
        return x