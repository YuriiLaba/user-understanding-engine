def resolve_torch_device(preferred: str | None = "auto") -> str:
    """Resolve a torch device, preferring CUDA when available."""
    if preferred and preferred != "auto":
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"
