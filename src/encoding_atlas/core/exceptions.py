"""Custom exceptions for encoding_atlas."""


class EncodingError(Exception):
    """Base exception for encoding errors."""
    pass


class ValidationError(EncodingError):
    """Raised when input validation fails."""
    pass


class BackendError(EncodingError):
    """Raised when backend operations fail."""
    pass


class RegistryError(EncodingError):
    """Raised when registry operations fail."""
    pass
