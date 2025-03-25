"""Custom exceptions for the project."""


class ModelLoadError(Exception):
    """Raised when there is an error loading the model."""

    pass


class TokenizerLoadError(Exception):
    """Raised when there is an error loading the tokenizer."""

    pass


class DatasetLoadError(Exception):
    """Raised when there is an error loading the dataset."""

    pass


class ConfigurationError(Exception):
    """Raised when there is an error in the configuration."""

    pass


class ValidationError(Exception):
    """Raised when there is a validation error."""

    pass


class ResourceError(Exception):
    """Raised when there is an error with system resources."""

    pass


class TrainingError(Exception):
    """Raised when there is an error during training."""

    pass
