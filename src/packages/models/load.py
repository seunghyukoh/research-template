import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)

logger = logging.get_logger(__name__)


def load_tokenizer(
    tokenizer_name_or_path,
    pad_with_eos=True,
    **kwargs,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        resume_download=None,  # ! Set to 'None' since this option is deprecated
        **kwargs,
    )

    if tokenizer.pad_token is None:
        if pad_with_eos:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            assert (
                kwargs.get("pad_token", None) is not None
            ), "pad_token must be defined"
            tokenizer.add_special_tokens({"pad_token": kwargs.get("pad_token")})

    return tokenizer


def load_model(
    model_name_or_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    use_custom_model=False,
    attn_implementation="flash_attention_2",
    **kwargs,
) -> PreTrainedModel:
    model_cls = use_custom_model or AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        resume_download=None,  # ! Set to 'None' since this option is deprecated
        attn_implementation=attn_implementation,
        **kwargs,
    )

    return model


def load_model_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    model_kwargs={},
    tokenizer_kwargs={},
):
    model = load_model(
        model_name_or_path=model_name_or_path,
        **model_kwargs,
    )

    tokenizer = load_tokenizer(
        # If the tokenizer_name_or_path is not defined, use model_name_or_path
        tokenizer_name_or_path=tokenizer_name_or_path or model_name_or_path,
        **tokenizer_kwargs,
    )

    return model, tokenizer
