"""Data models for frame detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

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
