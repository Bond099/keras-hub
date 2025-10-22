import keras
from keras import ops
from keras_hub.src.layers.modeling.transformer_layer_utils import compute_causal_mask, merge_padding_and_attention_mask
from keras_hub.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_hub.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock
from keras_hub.mech_interp.hooks import HookMixin
import inspect
from keras_hub.src.utils.keras_utils import fused_attention_op_available, gpu_supports_fused_attention_op, running_on_gpu, running_on_tpu

class HookedCachedGemmaAttention(CachedGemmaAttention, HookMixin):
    def __init__(self, *args, **kwargs):
        CachedGemmaAttention.__init__(self, *args, **kwargs)
        HookMixin.__init__(self)
        self.hidden_dim = self.num_query_heads * self.head_dim
        self.softmax = keras.layers.Softmax(axis=-1)

    def call(self, x, attention_mask=None, cache=None, cache_update_index=0, training=False):
        query = self.query_dense(x)
        query = self._apply_hooks(query, 'attn_q')
        query = self._apply_rope(query, cache_update_index)
        query = self._apply_hooks(query, 'attn_rot_q')
        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.key_dense(x)
            key_update = self._apply_hooks(key_update, 'attn_k')
            key_update = self._apply_rope(key_update, cache_update_index)
            key_update = self._apply_hooks(key_update, 'attn_rot_k')
            value_update = self.value_dense(x)
            value_update = self._apply_hooks(value_update, 'attn_v')
            key = ops.slice_update(key_cache, [0, cache_update_index, 0, 0], key_update)
            value = ops.slice_update(value_cache, [0, cache_update_index, 0, 0], value_update)
            cache = ops.stack((key, value), axis=1)
        else:
            key = self.key_dense(x)
            key = self._apply_hooks(key, 'attn_k')
            key = self._apply_rope(key, cache_update_index)
            key = self._apply_hooks(key, 'attn_rot_k')
            value = self.value_dense(x)
            value = self._apply_hooks(value, 'attn_v')
        attention_vec = self._compute_attention(query, key, value, attention_mask, training, cache_update_index)
        attention_vec = self._apply_hooks(attention_vec, 'attn_z')
        attention_output = self.output_dense(attention_vec)
        no_attended_tokens = ops.all(ops.equal(attention_mask, 0), axis=-1, keepdims=True)
        attention_output = ops.where(no_attended_tokens, ops.zeros_like(attention_output), attention_output)
        if cache is not None:
            return attention_output, cache
        return attention_output

    def _compute_attention(self, q, k, v, attention_mask, training=False, cache_update_index=0):
        if self.query_head_dim_normalize:
            query_normalization = 1 / ops.sqrt(ops.cast(self.head_dim, dtype=q.dtype))
        else:
            query_normalization = 1 / ops.sqrt(ops.cast(self.hidden_dim // self.num_query_heads, dtype=q.dtype))
        if self.use_sliding_window_attention and attention_mask is not None:
            attention_mask = self._mask_sliding_window(attention_mask, cache_update_index=cache_update_index)
        use_fused = self._use_fused_attention_op()
        if use_fused:
            kwargs = {"attn_logits_soft_cap": self.logit_soft_cap} if self.logit_soft_cap is not None else {}
            mask = ops.cast(ops.expand_dims(attention_mask, axis=1), "bool") if attention_mask is not None else None
            return ops.dot_product_attention(query=q, key=k, value=v, mask=mask, scale=query_normalization, **kwargs)
        q *= ops.cast(query_normalization, dtype=q.dtype)
        q_shape = ops.shape(q)
        q = ops.reshape(q, (*q_shape[:-2], self.num_key_value_heads, self.num_query_heads // self.num_key_value_heads, self.head_dim))
        k = ops.reshape(k, (*ops.shape(k)[:-2], self.num_key_value_heads, self.head_dim))
        v = ops.reshape(v, (*ops.shape(v)[:-2], self.num_key_value_heads, self.head_dim))
        b, q_len = ops.shape(q)[:2]
        h = ops.shape(q)[-1]
        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        attention_logits = self._apply_hooks(attention_logits, 'attn_scores')
        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(ops.tanh(attention_logits), self.logit_soft_cap)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :, :]
            attention_logits = ops.where(attention_mask == 0, -1e9, attention_logits)
        orig_dtype = attention_logits.dtype
        attention_logits = ops.cast(attention_logits, "float32")
        attention_softmax = self.softmax(attention_logits)
        attention_softmax = self._apply_hooks(attention_softmax, 'attn_pattern')
        attention_softmax = ops.cast(attention_softmax, orig_dtype)
        if self.dropout > 0:
            attention_softmax = self.dropout_layer(attention_softmax, training=training)
        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))

    def _use_fused_attention_op(self):
        if not fused_attention_op_available():
            return False
        if self.dropout > 0.0:
            return False
        if self.num_key_value_heads != self.num_query_heads:
            return False  # Skip fused for GQA to avoid shape issues
        if running_on_gpu():
            if self.logit_soft_cap is not None:
                return False
            return gpu_supports_fused_attention_op()
        elif running_on_tpu():
            sig = inspect.signature(ops.dot_product_attention)
            return "attn_logits_soft_cap" in sig.parameters
        else:
            return False

class HookedGemmaDecoderBlock(GemmaDecoderBlock, HookMixin):
    def __init__(self, *args, **kwargs):
        GemmaDecoderBlock.__init__(self, *args, **kwargs)
        HookMixin.__init__(self)
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
            name="attention"
        )

    def call(self, x, padding_mask=None, cache=None, cache_update_index=0):
        batch_size = ops.shape(x)[0]
        output_length = ops.shape(x)[1]
        input_length = output_length if cache is None else ops.shape(cache)[2]
        causal_mask = compute_causal_mask(batch_size=batch_size, input_length=input_length, output_length=output_length, cache_index=cache_update_index)
        attention_mask = merge_padding_and_attention_mask(x, padding_mask, causal_mask)
        x = self._apply_hooks(x, 'resid_pre')
        normalized_x = self.pre_attention_norm(x)
        normalized_x = self._apply_hooks(normalized_x, 'ln0')
        if cache is not None:
            attention_output, cache = self.attention(normalized_x, attention_mask=attention_mask, cache=cache, cache_update_index=cache_update_index)
        else:
            attention_output = self.attention(normalized_x, attention_mask=attention_mask)
        attention_output = self._apply_hooks(attention_output, 'attn_out')
        x = x + attention_output
        x = self._apply_hooks(x, 'resid_mid')
        if self.use_post_attention_norm:
            x = self.post_attention_norm(x)
        x = self._apply_hooks(x, 'ln1')
        normalized_x = self.pre_ffw_norm(x)
        normalized_x = self._apply_hooks(normalized_x, 'ln2')
        x1 = self.gating_ffw(normalized_x)
        x1 = self._apply_hooks(x1, 'mlp_gated1')
        x2 = self.gating_ffw_2(normalized_x)
        x2 = self._apply_hooks(x2, 'mlp_gated2')
        ffw_out = keras.activations.gelu(x1, approximate=True) * x2
        ffw_out = self.ffw_linear(ffw_out)
        ffw_out = self._apply_hooks(ffw_out, 'mlp_out')
        x = x + ffw_out
        if self.use_post_ffw_norm:
            x = self.post_ffw_norm(x)
        x = self._apply_hooks(x, 'ln3')
        x = self._apply_hooks(x, 'resid_post')
        if cache is not None:
            return x, cache
        return x

def hook_layer(original_layer, backbone):
    hooked_layer = HookedGemmaDecoderBlock(
        backbone.hidden_dim,
        backbone.intermediate_dim,
        backbone.head_dim,
        backbone.num_query_heads,
        backbone.num_key_value_heads,
        query_head_dim_normalize=backbone.query_head_dim_normalize,
        use_post_ffw_norm=backbone.use_post_ffw_norm,
        use_post_attention_norm=backbone.use_post_attention_norm,
        logit_soft_cap=backbone.attention_logit_soft_cap,
        use_sliding_window_attention=backbone.use_sliding_window_attention,
        sliding_window_size=backbone.sliding_window_size,
        layer_norm_epsilon=backbone.layer_norm_epsilon,
        dropout=backbone.dropout,
    )
    dummy_shape = (1, 4, backbone.hidden_dim)
    dummy_x = ops.zeros(dummy_shape, dtype=backbone.dtype_policy.compute_dtype)
    dummy_padding_mask = ops.ones((1, 4), dtype="int32")
    hooked_layer(dummy_x, padding_mask=dummy_padding_mask)
    hooked_layer.pre_attention_norm.set_weights(original_layer.pre_attention_norm.get_weights())
    hooked_layer.pre_ffw_norm.set_weights(original_layer.pre_ffw_norm.get_weights())
    hooked_layer.attention.set_weights(original_layer.attention.get_weights())
    hooked_layer.gating_ffw.set_weights(original_layer.gating_ffw.get_weights())
    hooked_layer.gating_ffw_2.set_weights(original_layer.gating_ffw_2.get_weights())
    hooked_layer.ffw_linear.set_weights(original_layer.ffw_linear.get_weights())
    if hasattr(original_layer, 'post_attention_norm'):
        hooked_layer.post_attention_norm.set_weights(original_layer.post_attention_norm.get_weights())
    if hasattr(original_layer, 'post_ffw_norm'):
        hooked_layer.post_ffw_norm.set_weights(original_layer.post_ffw_norm.get_weights())
    return hooked_layer