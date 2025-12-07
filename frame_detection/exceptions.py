"""Custom exceptions for frame detection."""


class FrameDetectionError(Exception):
    """Base exception for frame detection errors."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class ImageReadError(FrameDetectionError):
    """Failed to read input image."""

    def __init__(self, path: str):
        super().__init__(
            f"Could not read image: {path}",
            "Could not read image file. The file may be corrupted or in an unsupported format.",
        )


class NoLinesDetectedError(FrameDetectionError):
    """No edge lines detected in the image."""

    def __init__(self):
        super().__init__(
            "No lines detected",
            "Could not detect frame edges. Try adjusting the edge margin or separation method settings.",
        )


class NoFrameEdgesError(FrameDetectionError):
    """Lines detected but couldn't classify into frame edges."""

    def __init__(self):
        super().__init__(
            "No frame edges detected",
            "Could not classify frame edges. The image may not contain a clear film frame.",
        )


class InvalidFrameRegionError(FrameDetectionError):
    """Detected frame region has invalid bounds."""

    def __init__(self, detail: str = ""):
        msg = f"Invalid frame region detected: {detail}" if detail else "Invalid frame region detected"
        super().__init__(
            msg,
            "Detected frame region is invalid. The image may have unusual content or require different settings.",
        )


class InvalidFrameCoordinatesError(FrameDetectionError):
    """Final frame coordinates are invalid."""

    def __init__(self):
        super().__init__(
            "Invalid frame coordinates detected",
            "Could not determine valid crop coordinates. Try different settings.",
        )
