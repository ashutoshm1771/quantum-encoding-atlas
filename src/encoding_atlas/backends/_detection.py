"""Backend availability detection."""


def is_pennylane_available() -> bool:
    """Check if PennyLane is installed."""
    try:
        import pennylane
        return True
    except ImportError:
        return False


def is_qiskit_available() -> bool:
    """Check if Qiskit is installed."""
    try:
        import qiskit
        return True
    except ImportError:
        return False


def is_cirq_available() -> bool:
    """Check if Cirq is installed."""
    try:
        import cirq
        return True
    except ImportError:
        return False


def get_available_backends() -> list[str]:
    """Get list of available backends."""
    backends = []
    if is_pennylane_available():
        backends.append("pennylane")
    if is_qiskit_available():
        backends.append("qiskit")
    if is_cirq_available():
        backends.append("cirq")
    return backends
