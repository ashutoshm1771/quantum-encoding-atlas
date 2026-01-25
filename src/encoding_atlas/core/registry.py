"""Encoding registry system."""

from typing import TYPE_CHECKING, Any, Type

from encoding_atlas.core.exceptions import RegistryError

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

_ENCODING_REGISTRY: dict[str, Type["BaseEncoding"]] = {}


def register_encoding(name: str) -> Any:
    """Decorator to register an encoding class.

    Parameters
    ----------
    name : str
        Name to register the encoding under.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @register_encoding("my_encoding")
    ... class MyEncoding(BaseEncoding):
    ...     pass
    """

    def decorator(cls: Type["BaseEncoding"]) -> Type["BaseEncoding"]:
        if name in _ENCODING_REGISTRY:
            raise RegistryError(f"Encoding '{name}' is already registered")
        _ENCODING_REGISTRY[name] = cls
        return cls

    return decorator


def get_encoding(name: str, **kwargs: Any) -> "BaseEncoding":
    """Get an encoding instance by name.

    Parameters
    ----------
    name : str
        Name of the registered encoding.
    **kwargs : Any
        Parameters to pass to the encoding constructor.

    Returns
    -------
    BaseEncoding
        Instantiated encoding.

    Raises
    ------
    RegistryError
        If encoding name is not registered.
    """
    if name not in _ENCODING_REGISTRY:
        available = ", ".join(sorted(_ENCODING_REGISTRY.keys()))
        raise RegistryError(
            f"Unknown encoding '{name}'. Available encodings: {available}"
        )
    return _ENCODING_REGISTRY[name](**kwargs)


def list_encodings() -> list[str]:
    """List all registered encoding names.

    Returns
    -------
    list[str]
        Sorted list of encoding names.
    """
    return sorted(_ENCODING_REGISTRY.keys())


def _clear_registry() -> None:
    """Clear the registry. For testing only."""
    _ENCODING_REGISTRY.clear()
