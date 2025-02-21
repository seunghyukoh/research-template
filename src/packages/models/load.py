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
        **kwargs,
    )

    if tokenizer.pad_token is None:
        if pad_with_eos:
            logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            assert kwargs.get("pad_token", None) is not None, (
                "pad_token must be defined"
            )
            logger.info("Setting pad_token to %s", kwargs.get("pad_token"))
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
    device_map = device_map or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()  # Apple Silicon
        else "cpu"
    )

    model_cls = use_custom_model or AutoModelForCausalLM

    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning("FlashAttention is not installed.")
            attn_implementation = "eager"

    model = model_cls.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
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
