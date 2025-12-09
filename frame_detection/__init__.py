"""Frame detection and cropping for film negatives."""

__version__ = "0.0.9"


def __getattr__(name):
    """Lazy import to avoid loading cv2 for CLI subcommands that don't need it."""
    if name in ("crop_frame", "detect_frame_bounds"):
        from .detection import crop_frame, detect_frame_bounds
        return {"crop_frame": crop_frame, "detect_frame_bounds": detect_frame_bounds}[name]
    if name in ("EdgeGroup", "FrameBounds", "Line", "Margins"):
        from .models import EdgeGroup, FrameBounds, Line, Margins
        return {"EdgeGroup": EdgeGroup, "FrameBounds": FrameBounds, "Line": Line, "Margins": Margins}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "detect_frame_bounds",
    "crop_frame",
    "Margins",
    "Line",
    "EdgeGroup",
    "FrameBounds",
]
