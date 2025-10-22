import numpy as np

def get_act_name(act_type, layer_idx=None, head_idx=None):
    base = f"blocks.{layer_idx}." if layer_idx is not None else ""
    if head_idx is not None:
        base += f"head.{head_idx}."
    return base + f"hook_{act_type}"

def to_tokens(inputs, preprocessor):
    preprocessed = preprocessor.generate_preprocess(inputs)
    return preprocessed['token_ids']

def to_str_tokens(tokens, preprocessor):
    postprocessed = preprocessor.generate_postprocess({'token_ids': tokens})
    return postprocessed  # Assume strings