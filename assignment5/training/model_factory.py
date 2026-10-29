"""HF policy/tokenizer construction independent of trainer lifecycle."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_id, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(model_id, *, dtype, device, local_files_only=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, local_files_only=local_files_only).to(device)
    tokenizer = load_tokenizer(model_id, local_files_only=local_files_only)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def policy_factory(model_id, pad_token_id, *, disable_dropout=False, max_sequence_length=None):
    def build():
        config = AutoConfig.from_pretrained(model_id)
        if disable_dropout:
            for cfg in (config, getattr(config, "text_config", config)):
                for name in ("attention_dropout", "attention_probs_dropout_prob", "hidden_dropout_prob",
                             "resid_pdrop", "embd_pdrop", "attn_pdrop"):
                    if hasattr(cfg, name):
                        setattr(cfg, name, 0.0)
        config.pad_token_id = pad_token_id
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=torch.float32)
        limit = getattr(model.config, "max_position_embeddings", None)
        if limit and max_sequence_length is not None and max_sequence_length > limit:
            raise ValueError("SFT sequences exceed the model context limit")
        return model
    return build




def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
