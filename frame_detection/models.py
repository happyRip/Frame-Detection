"""Data models for frame detection."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np


class Orientation(Enum):
    """Orientation of sprocket holes on the film."""

    HORIZONTAL = "horizontal"  # Sprockets on top/bottom edges
    VERTICAL = "vertical"  # Sprockets on left/right edges


class FilmType(Enum):
    """Type of film stock affecting detection strategy.

    NEGATIVE: Sprocket holes appear bright (light passes through during scan)
    POSITIVE: Sprocket holes appear dark (slide film or dark background scan)
    AUTO: Automatically detect based on histogram analysis
    """

    NEGATIVE = "negative"
    POSITIVE = "positive"
    AUTO = "auto"


class SprocketType(Enum):
    """Type of sprocket holes expected in the image.

    NONE: Skip sprocket detection entirely (medium format, pre-cropped images)
    BRIGHT: Sprocket holes appear bright (light passes through during scan)
    DARK: Sprocket holes appear dark (opaque during scan)
    AUTO: Automatically detect based on histogram analysis
    """

    NONE = "none"
    BRIGHT = "bright"
    DARK = "dark"
    AUTO = "auto"


@dataclass
class Margins:
    """Represents margins for edges (top, right, bottom, left)."""

    top: float
    right: float
    bottom: float
    left: float

    @classmethod
    def parse(cls, value: str) -> Margins:
        """Parse margin string into Margins object.

        Supports formats:
            - Single value: "30" -> all edges 30%
            - Two values: "30,40" -> vertical 30%, horizontal 40%
            - Four values: "30,40,50,10" -> top, right, bottom, left

        Separators: , : / ;

        Returns:
            Margins object with values as fractions (0-1)
        """
        parts = re.split(r"[,:;/]", value)
        parts = [float(p) / 100 for p in parts]

        if len(parts) == 1:
            return cls(parts[0], parts[0], parts[0], parts[0])
        elif len(parts) == 2:
            # vertical, horizontal
            return cls(parts[0], parts[1], parts[0], parts[1])
        elif len(parts) == 4:
            # top, right, bottom, left
            return cls(parts[0], parts[1], parts[2], parts[3])
        else:
            raise ValueError(f"Invalid margin format: {value}")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return margins as (top, right, bottom, left) tuple."""
        return (self.top, self.right, self.bottom, self.left)


@dataclass
class Line:
    """Represents a detected line segment."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_hough(cls, line_data: np.ndarray) -> Line:
        """Create Line from HoughLinesP output format."""
        x1, y1, x2, y2 = line_data
        return cls(int(x1), int(y1), int(x2), int(y2))

    @property
    def angle(self) -> float:
        """Return angle in degrees (-90 to 90)."""
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def length(self) -> float:
        """Return line length in pixels."""
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def midpoint(self) -> tuple[float, float]:
        """Return midpoint (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def avg_x(self) -> float:
        """Return average x position."""
        return (self.x1 + self.x2) / 2

    @property
    def avg_y(self) -> float:
        """Return average y position."""
        return (self.y1 + self.y2) / 2

    @property
    def is_horizontal(self) -> bool:
        """Check if line is near horizontal (within 20 degrees)."""
        dx = abs(self.x2 - self.x1)
        dy = abs(self.y2 - self.y1)
        slope_threshold = 0.36  # tan(20°)
        return dy <= dx * slope_threshold

    @property
    def is_vertical(self) -> bool:
        """Check if line is near vertical (within 20 degrees)."""
        dx = abs(self.x2 - self.x1)
        dy = abs(self.y2 - self.y1)
        slope_threshold = 0.36  # tan(20°)
        return dx <= dy * slope_threshold

    def offset_y(self, delta: int) -> Line:
        """Return new line with y coordinates offset by delta."""
        return Line(self.x1, self.y1 + delta, self.x2, self.y2 + delta)

    def offset_x(self, delta: int) -> Line:
        """Return new line with x coordinates offset by delta."""
        return Line(self.x1 + delta, self.y1, self.x2 + delta, self.y2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return line as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


EdgeType = Literal["top", "bottom", "left", "right"]


@dataclass
class EdgeGroup:
    """Collection of lines belonging to one edge of the frame."""

    edge: EdgeType
    lines: list[Line] = field(default_factory=list)

    def add(self, line: Line) -> None:
        """Add a line to this edge group."""
        self.lines.append(line)

    @property
    def average_position(self) -> float | None:
        """Return average position (y for horizontal edges, x for vertical)."""
        if not self.lines:
            return None

        if self.edge in ("top", "bottom"):
            return sum(line.avg_y for line in self.lines) / len(self.lines)
        else:
            return sum(line.avg_x for line in self.lines) / len(self.lines)

    @property
    def min_position(self) -> float | None:
        """Return minimum position."""
        if not self.lines:
            return None

        if self.edge in ("top", "bottom"):
            positions = [line.y1 for line in self.lines] + [line.y2 for line in self.lines]
        else:
            positions = [line.x1 for line in self.lines] + [line.x2 for line in self.lines]
        return min(positions)

    @property
    def max_position(self) -> float | None:
        """Return maximum position."""
        if not self.lines:
            return None

        if self.edge in ("top", "bottom"):
            positions = [line.y1 for line in self.lines] + [line.y2 for line in self.lines]
        else:
            positions = [line.x1 for line in self.lines] + [line.x2 for line in self.lines]
        return max(positions)


@dataclass
class FilmCutEnd:
    """Represents detection of film cut ends visible in the viewport.

    When a film strip has been cut, the cut end may be visible as a large
    bright area at the edge of the viewport (no film covering that area).
    In landscape orientation, this appears on left/right edges.
    """

    left: bool = False
    right: bool = False
    top: bool = False
    bottom: bool = False

    @property
    def any_detected(self) -> bool:
        """Check if any cut end is detected."""
        return self.left or self.right or self.top or self.bottom


@dataclass
class FilmBaseResult:
    """Result of film base color detection with variance analysis.

    Contains the detected film base color along with variance statistics
    used for adaptive tolerance calculation.
    """

    color: np.ndarray
    """BGR median color of the film base."""

    variance: float
    """Combined color variance (Euclidean norm of channel stddevs)."""

    iqr: np.ndarray
    """Per-channel IQR (interquartile range) [B, G, R]."""

    sample_count: int
    """Number of samples used after outlier rejection."""

    suggested_tolerance: int
    """Adaptive tolerance based on variance."""

    outlier_mask: np.ndarray
    """Mask of filtered outlier pixels (large regions only, no dust)."""


@dataclass
class FrameBounds:
    """Represents detected frame boundaries."""

    top: EdgeGroup = field(default_factory=lambda: EdgeGroup("top"))
    bottom: EdgeGroup = field(default_factory=lambda: EdgeGroup("bottom"))
    left: EdgeGroup = field(default_factory=lambda: EdgeGroup("left"))
    right: EdgeGroup = field(default_factory=lambda: EdgeGroup("right"))

    def as_rectangle(self) -> tuple[int, int, int, int] | None:
        """Return axis-aligned bounding rectangle as (left, right, top, bottom).

        Uses min/max positions from edge groups.
        """
        left_pos = self.left.min_position
        right_pos = self.right.max_position
        top_pos = self.top.min_position
        bottom_pos = self.bottom.max_position

        if any(p is None for p in [left_pos, right_pos, top_pos, bottom_pos]):
            return None

        return (int(left_pos), int(right_pos), int(top_pos), int(bottom_pos))

    @property
    def has_all_edges(self) -> bool:
        """Check if all four edges have detected lines."""
        return all([
            self.top.lines,
            self.bottom.lines,
            self.left.lines,
            self.right.lines,
        ])


# =============================================================================
# Filter Configuration Classes
# =============================================================================


@dataclass
class CannyParams:
    """Parameters for Canny edge detection."""

    low: int = 50
    high: int = 150

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (0 <= self.low <= 255):
            raise ValueError(f"canny.low must be 0-255, got {self.low}")
        if not (0 <= self.high <= 255):
            raise ValueError(f"canny.high must be 0-255, got {self.high}")
        if self.low >= self.high:
            raise ValueError(f"canny.low ({self.low}) must be < canny.high ({self.high})")


@dataclass
class BlurParams:
    """Parameters for filters using Gaussian blur (Sobel, Scharr, Laplacian)."""

    blur_size: int = 5

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (0 <= self.blur_size <= 21):
            raise ValueError(f"blur_size must be 0-21, got {self.blur_size}")
        if self.blur_size > 0 and self.blur_size % 2 == 0:
            raise ValueError(f"blur_size must be 0 or odd, got {self.blur_size}")


@dataclass
class DoGParams:
    """Parameters for Difference of Gaussians filter."""

    sigma1: float = 1.0
    sigma2: float = 2.0

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (0.1 <= self.sigma1 <= 5.0):
            raise ValueError(f"dog.sigma1 must be 0.1-5.0, got {self.sigma1}")
        if not (0.1 <= self.sigma2 <= 10.0):
            raise ValueError(f"dog.sigma2 must be 0.1-10.0, got {self.sigma2}")


@dataclass
class LoGParams:
    """Parameters for Laplacian of Gaussian filter."""

    sigma: float = 2.0

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (0.1 <= self.sigma <= 5.0):
            raise ValueError(f"log.sigma must be 0.1-5.0, got {self.sigma}")


@dataclass
class EdgeFilterConfig:
    """Configuration for edge detection filters."""

    method: str = "scharr"
    canny: CannyParams = field(default_factory=CannyParams)
    sobel: BlurParams = field(default_factory=BlurParams)
    scharr: BlurParams = field(default_factory=BlurParams)
    laplacian: BlurParams = field(default_factory=BlurParams)
    dog: DoGParams = field(default_factory=DoGParams)
    log: LoGParams = field(default_factory=LoGParams)

    def validate(self) -> None:
        """Validate the configuration."""
        valid_methods = {"canny", "sobel", "scharr", "dog", "laplacian", "log"}
        if self.method not in valid_methods:
            raise ValueError(f"edge_filter.method must be one of {valid_methods}")
        # Validate the selected method's params
        getattr(self, self.method).validate()

    def get_params(self) -> dict[str, Any]:
        """Get parameters for the selected method."""
        return asdict(getattr(self, self.method))


@dataclass
class ClaheParams:
    """Parameters for CLAHE separation method."""

    clip_limit: float = 1.0
    tile_size: int = 32

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (0.1 <= self.clip_limit <= 10.0):
            raise ValueError(f"clahe.clip_limit must be 0.1-10.0, got {self.clip_limit}")
        if not (8 <= self.tile_size <= 128):
            raise ValueError(f"clahe.tile_size must be 8-128, got {self.tile_size}")


@dataclass
class AdaptiveParams:
    """Parameters for adaptive separation method."""

    block_size: int = 51

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (11 <= self.block_size <= 201):
            raise ValueError(f"adaptive.block_size must be 11-201, got {self.block_size}")
        if self.block_size % 2 == 0:
            raise ValueError(f"adaptive.block_size must be odd, got {self.block_size}")


@dataclass
class GradientSepParams:
    """Parameters for gradient separation method."""

    gradient_weight: float = 0.5

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not (0.0 <= self.gradient_weight <= 1.0):
            raise ValueError(
                f"gradient.gradient_weight must be 0.0-1.0, got {self.gradient_weight}"
            )


@dataclass
class SeparationConfig:
    """Configuration for film base separation methods."""

    method: str = "color_distance"
    tolerance: int = 30
    adaptive_min: int = 10
    adaptive_max: int = 30
    gradient_tolerance: bool = False
    clahe: ClaheParams = field(default_factory=ClaheParams)
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)
    gradient: GradientSepParams = field(default_factory=GradientSepParams)

    def validate(self) -> None:
        """Validate the configuration."""
        valid_methods = {
            "color_distance",
            "clahe",
            "lab_distance",
            "hsv_distance",
            "adaptive",
            "gradient",
        }
        if self.method not in valid_methods:
            raise ValueError(f"separation.method must be one of {valid_methods}")
        if not (0 <= self.tolerance <= 255):
            raise ValueError(f"separation.tolerance must be 0-255, got {self.tolerance}")
        if not (1 <= self.adaptive_min <= 100):
            raise ValueError(f"separation.adaptive_min must be 1-100, got {self.adaptive_min}")
        if not (1 <= self.adaptive_max <= 100):
            raise ValueError(f"separation.adaptive_max must be 1-100, got {self.adaptive_max}")
        if self.adaptive_min > self.adaptive_max:
            raise ValueError(
                f"separation.adaptive_min ({self.adaptive_min}) must be <= "
                f"adaptive_max ({self.adaptive_max})"
            )
        # Validate method-specific params if applicable
        if self.method in ("clahe", "adaptive", "gradient"):
            getattr(self, self.method).validate()

    def get_params(self) -> dict[str, Any]:
        """Get parameters for the selected method, including tolerance."""
        params: dict[str, Any] = {
            "tolerance": self.tolerance,
            "adaptive_min": self.adaptive_min,
            "adaptive_max": self.adaptive_max,
            "gradient_tolerance": self.gradient_tolerance,
        }
        if self.method in ("clahe", "adaptive", "gradient"):
            params.update(asdict(getattr(self, self.method)))
        return params


@dataclass
class FilterConfig:
    """Complete filter configuration for frame detection."""

    edge_filter: EdgeFilterConfig = field(default_factory=EdgeFilterConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)

    def validate(self) -> None:
        """Validate all configuration."""
        self.edge_filter.validate()
        self.separation.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterConfig:
        """Create FilterConfig from dictionary."""
        config = cls()

        if "edge_filter" in data:
            ef = data["edge_filter"]
            if "method" in ef:
                config.edge_filter.method = ef["method"]
            for method in ["canny", "sobel", "scharr", "laplacian", "dog", "log"]:
                if method in ef:
                    for key, value in ef[method].items():
                        setattr(getattr(config.edge_filter, method), key, value)

        if "separation" in data:
            sep = data["separation"]
            if "method" in sep:
                config.separation.method = sep["method"]
            if "tolerance" in sep:
                config.separation.tolerance = sep["tolerance"]
            if "adaptive_min" in sep:
                config.separation.adaptive_min = sep["adaptive_min"]
            if "adaptive_max" in sep:
                config.separation.adaptive_max = sep["adaptive_max"]
            if "gradient_tolerance" in sep:
                config.separation.gradient_tolerance = sep["gradient_tolerance"]
            for method in ["clahe", "adaptive", "gradient"]:
                if method in sep:
                    for key, value in sep[method].items():
                        setattr(getattr(config.separation, method), key, value)

        config.validate()
        return config

    @classmethod
    def from_json(cls, json_str: str) -> FilterConfig:
        """Parse FilterConfig from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> FilterConfig:
        """Load FilterConfig from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())

    @classmethod
    def default_json(cls) -> str:
        """Return default configuration as formatted JSON string."""
        config = cls()
        return config.to_json()
