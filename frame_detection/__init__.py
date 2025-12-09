"""Frame detection and cropping for film negatives."""

from .detection import crop_frame, detect_frame_bounds
from .models import EdgeGroup, FrameBounds, Line, Margins

__version__ = "0.0.3"
__all__ = [
    "detect_frame_bounds",
    "crop_frame",
    "Margins",
    "Line",
    "EdgeGroup",
    "FrameBounds",
]
