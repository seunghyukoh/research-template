"""Dataset preprocessing functions registry.

This module provides a registry pattern for dataset preprocessing functions,
making it easy to add new datasets without modifying the main training code.

Usage:
    # Register a new preprocessing function
    @register_preprocessing_fn("my_dataset")
    def preprocess_my_dataset(example):
        return {"prompt": example["input"], "completion": example["output"]}

    # Get a preprocessing function
    fn = get_preprocessing_fn("my_dataset")
    processed = fn(example)
"""

from typing import Any, Callable, Dict

# Registry to store preprocessing functions
_PREPROCESSING_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, str]]] = {}


def register_preprocessing_fn(name: str):
    """Decorator to register a preprocessing function.

    Args:
        name: The name to register the function under (e.g., dataset name or identifier)

    Returns:
        The decorator function

    Example:
        @register_preprocessing_fn("gsm8k")
        def preprocess_gsm8k(example):
            return {"prompt": example["question"], "completion": example["answer"]}
    """

    def decorator(fn: Callable[[Dict[str, Any]], Dict[str, str]]):
        if name in _PREPROCESSING_REGISTRY:
            raise ValueError(
                f"Preprocessing function '{name}' is already registered. "
                f"Use a different name or unregister the existing function."
            )
        _PREPROCESSING_REGISTRY[name] = fn
        return fn

    return decorator


def get_preprocessing_fn(name: str) -> Callable[[Dict[str, Any]], Dict[str, str]]:
    """Get a registered preprocessing function by name.

    Args:
        name: The name of the preprocessing function to retrieve

    Returns:
        The preprocessing function

    Raises:
        ValueError: If the preprocessing function is not registered
    """
    if name not in _PREPROCESSING_REGISTRY:
        available = ", ".join(sorted(_PREPROCESSING_REGISTRY.keys()))
        raise ValueError(
            f"Preprocessing function '{name}' not found. " f"Available functions: {available}"
        )
    return _PREPROCESSING_REGISTRY[name]


def list_preprocessing_fns() -> list[str]:
    """List all registered preprocessing function names.

    Returns:
        A sorted list of registered preprocessing function names
    """
    return sorted(_PREPROCESSING_REGISTRY.keys())


# ============================================================================
# Built-in preprocessing functions
# ============================================================================


@register_preprocessing_fn("gsm8k")
def preprocess_gsm8k(example: Dict[str, Any]) -> Dict[str, str]:
    """Preprocess GSM8K dataset examples.

    GSM8K (Grade School Math 8K) is a dataset of math word problems.

    Args:
        example: A dictionary containing "question" and "answer" fields

    Returns:
        A dictionary with "prompt" and "completion" fields
    """
    return {
        "prompt": example["question"],
        "completion": example["answer"],
    }
