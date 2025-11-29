"""Frame detection and cropping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from debug_visualizer import DebugVisualizer

# Edge indices for bounds and certainties arrays
LEFT, RIGHT, TOP, BOTTOM = 0, 1, 2, 3


def detect_sprocket_holes(
    img: np.ndarray, visualizer: DebugVisualizer | None = None
) -> np.ndarray:
    """Detect sprocket holes as the rightmost peak in the tone curve.

    Sprocket holes appear as overexposed white regions. In the histogram,
    they form a distinct peak at the far right (bright end).

    Args:
        img: Input image as numpy array
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Binary mask where sprocket holes are marked as 255
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Smooth histogram to reduce noise
    kernel_size = 5
    hist_smooth = np.convolve(hist, np.ones(kernel_size) / kernel_size, mode="same")

    # Find peaks in the bright region (right side of histogram, 200-255)
    bright_region = hist_smooth[200:]
    sprocket_mask = np.zeros(gray.shape, dtype=np.uint8)
    valley_idx = None
    threshold_idx = None

    if bright_region.max() >= hist_smooth.max() * 0.02:
        # Find the rightmost significant peak
        threshold = bright_region.max() * 0.3
        peak_idx = None
        for i in range(len(bright_region) - 1, 0, -1):
            if bright_region[i] > threshold:
                peak_idx = 200 + i
                break

        if peak_idx is not None and peak_idx >= 210:
            # Find the valley (minimum) before the peak
            search_start = max(0, peak_idx - 200 - 30)  # Look up to 30 bins left of 200
            search_end = peak_idx - 200
            valley_idx = (
                200 + np.argmin(bright_region[search_start:search_end]) + search_start
            )

            # Find the steepest rise between valley and peak (max derivative)
            region = hist_smooth[valley_idx:peak_idx]
            if len(region) > 1:
                derivative = np.diff(region)
                steepest_offset = np.argmax(derivative)
                # Threshold just before the steepest rise
                threshold_idx = valley_idx + steepest_offset
            else:
                threshold_idx = valley_idx

            sprocket_mask = (gray >= threshold_idx).astype(np.uint8) * 255

            # Dilate to ensure we cover the hole edges
            dilate_k = max(3, int(min(gray.shape[:2]) / 150) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            sprocket_mask = cv2.dilate(sprocket_mask, kernel, iterations=2)

    if visualizer:
        visualizer.save_sprocket_holes(img, sprocket_mask, hist_smooth, threshold_idx)

    return sprocket_mask


def crop_sprocket_region(
    img: np.ndarray,
    sprocket_mask: np.ndarray,
    visualizer: DebugVisualizer | None = None,
) -> tuple[np.ndarray, int, int]:
    """Crop out the region containing sprocket holes.

    Sprocket holes on 35mm film run along both long edges (top and bottom).
    This finds the sprocket regions and crops them from both edges.

    Args:
        img: Input image as numpy array
        sprocket_mask: Binary mask where sprocket holes are marked as 255
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Tuple of (cropped_image, y_offset_top, y_offset_bottom) where offsets
        indicate pixels cropped from top and bottom edges.
    """
    if sprocket_mask.max() == 0:
        if visualizer:
            visualizer.save_sprocket_crop(img, img, 0, 0)
        return img, 0, 0

    img_h, img_w = img.shape[:2]
    mid_y = img_h // 2

    # Find rows containing sprocket pixels
    rows_with_sprockets = np.any(sprocket_mask > 0, axis=1)

    if not np.any(rows_with_sprockets):
        if visualizer:
            visualizer.save_sprocket_crop(img, img, 0, 0)
        return img, 0, 0

    # Find crop boundaries for top and bottom
    sprocket_rows = np.where(rows_with_sprockets)[0]

    # Small margin relative to resolution
    margin = max(1, img_h // 500)

    # Top crop: find lowest sprocket row in top half
    top_sprockets = sprocket_rows[sprocket_rows < mid_y]
    if len(top_sprockets) > 0:
        crop_top = np.max(top_sprockets) + 1
        crop_top = min(crop_top + margin, mid_y)
    else:
        crop_top = 0

    # Bottom crop: find highest sprocket row in bottom half
    bottom_sprockets = sprocket_rows[sprocket_rows >= mid_y]
    if len(bottom_sprockets) > 0:
        crop_bottom = np.min(bottom_sprockets)
        crop_bottom = max(crop_bottom - margin, mid_y)
    else:
        crop_bottom = img_h

    img_cropped = img[crop_top:crop_bottom, :]

    if visualizer:
        visualizer.save_sprocket_crop(img, img_cropped, crop_top, img_h - crop_bottom)

    return img_cropped, crop_top, img_h - crop_bottom


def detect_lines(
    img: np.ndarray, visualizer: DebugVisualizer | None = None
) -> tuple[np.ndarray | None, int, int]:
    """Detect lines in an image using Canny edge detection and Hough transform.

    Internally crops sprocket hole regions before edge detection, then adjusts
    line coordinates back to original image space.

    Returns:
        Tuple of (lines, y_min, y_max) where y_min and y_max define the valid
        vertical region excluding sprocket holes.
    """
    # Detect sprocket holes and crop them out before edge detection
    sprocket_mask = detect_sprocket_holes(img, visualizer)
    img_cropped, y_offset_top, y_offset_bottom = crop_sprocket_region(
        img, sprocket_mask, visualizer
    )

    img_h = img.shape[0]
    y_min = y_offset_top
    y_max = img_h - y_offset_bottom

    # Re-normalize after cropping out sprocket holes for better contrast
    img_cropped = normalize_levels(img_cropped, visualizer)

    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    k = max(
        3, int(min(gray.shape[:2]) / 200) | 1
    )  # Scale kernel with resolution, ensure odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    median = np.median(blurred)
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, low, high)

    if visualizer:
        visualizer.save_edges(edges)

    min_dim = min(img_cropped.shape[:2])
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_dim * 0.2,
        maxLineGap=min_dim // 10,
    )

    # Adjust line coordinates back to original image space
    if lines is not None and y_offset_top > 0:
        lines[:, :, 1] += y_offset_top  # y1
        lines[:, :, 3] += y_offset_top  # y2

    if visualizer and lines is not None:
        visualizer.save_lines(img, lines)

    return lines, y_min, y_max


def classify_lines(lines: np.ndarray) -> tuple[list[int], list[int]]:
    """Classify lines as horizontal or vertical based on slope.

    Returns:
        Tuple of (horizontal y-positions, vertical x-positions)
    """
    horiz_positions = []
    vert_positions = []

    # tan(20°) ≈ 0.36 - lines within 20° of axis are classified
    slope_threshold = 0.36

    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy <= dx * slope_threshold:  # near horizontal
            horiz_positions.extend([y1, y2])
        elif dx <= dy * slope_threshold:  # near vertical
            vert_positions.extend([x1, x2])

    return horiz_positions, vert_positions


def resolve_axis(
    positions: list[int],
    img_size: int,
    other_size: float,
    aspect_ratio: float,
    is_width: bool,
) -> tuple[int, int]:
    """Resolve two edge positions for an axis from detected positions.

    Returns:
        Tuple of (low_pos, high_pos)
    """
    if len(positions) >= 2:
        return int(np.min(positions)), int(np.max(positions))

    if len(positions) == 1:
        edge_pos = int(positions[0])
        size = (
            int(round(other_size * aspect_ratio))
            if is_width
            else int(round(other_size / aspect_ratio))
        )
        # Place edge as start or end based on which is closer to image boundary
        if edge_pos < img_size // 2:
            return edge_pos, edge_pos + size
        else:
            return edge_pos - size, edge_pos

    # No detection: center a rectangle estimated from other axis
    size = (
        int(round(other_size * aspect_ratio))
        if is_width
        else int(round(other_size / aspect_ratio))
    )
    return (img_size - size) // 2, (img_size + size) // 2


def normalize_levels(
    img: np.ndarray,
    visualizer: DebugVisualizer | None = None,
) -> np.ndarray:
    """Normalize image levels to utilize full exposure range.

    Stretches the histogram so that blacks are at 0 and whites at 255.
    Uses percentile-based clipping to avoid outliers affecting the range.

    Args:
        img: Input image as numpy array
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Normalized image as numpy array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use percentiles to find black and white points (avoid outliers)
    black_point = int(np.percentile(gray, 1))
    white_point = int(np.percentile(gray, 99))

    # Avoid division by zero
    if white_point <= black_point:
        white_point = black_point + 1

    # Normalize each channel
    img_normalized = img.astype(np.float32)
    img_normalized = (img_normalized - black_point) / (white_point - black_point)
    img_normalized = np.clip(img_normalized * 255, 0, 255).astype(np.uint8)

    if visualizer:
        visualizer.save_normalize_levels(img, img_normalized, black_point, white_point)

    return img_normalized


def apply_crop_in(
    img: np.ndarray,
    bounds: list[int],
    crop_in_percent: float,
    visualizer: DebugVisualizer | None = None,
) -> list[int]:
    """Shrink bounds inward by a percentage to exclude unexposed borders.

    Args:
        img: Input image (used for visualization)
        bounds: Array of [left, right, top, bottom] positions
        crop_in_percent: Percentage of each dimension to crop from edges (0-100)
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Adjusted bounds with crop-in applied
    """
    if crop_in_percent <= 0:
        return bounds

    bounds_before = bounds.copy()
    bounds = bounds.copy()
    width = bounds[RIGHT] - bounds[LEFT]
    height = bounds[BOTTOM] - bounds[TOP]

    crop_x = int(width * crop_in_percent / 100)
    crop_y = int(height * crop_in_percent / 100)

    bounds[LEFT] += crop_x
    bounds[RIGHT] -= crop_x
    bounds[TOP] += crop_y
    bounds[BOTTOM] -= crop_y

    if visualizer:
        visualizer.save_crop_in(img, bounds_before, bounds, crop_in_percent)

    return bounds


def enforce_aspect_ratio(
    bounds: list[int],
    aspect_ratio: float,
    y_min: int,
    y_max: int,
    img_w: int,
) -> list[int]:
    """Adjust bounds to enforce the correct aspect ratio.

    Shrinks the larger dimension to match the expected aspect ratio,
    keeping the crop centered.

    Args:
        bounds: Array of [left, right, top, bottom] positions
        aspect_ratio: Expected aspect ratio (width/height)
        y_min: Minimum valid y coordinate (sprocket boundary)
        y_max: Maximum valid y coordinate (sprocket boundary)
        img_w: Image width for clamping

    Returns:
        Adjusted bounds with correct aspect ratio
    """
    bounds = bounds.copy()
    width = bounds[RIGHT] - bounds[LEFT]
    height = bounds[BOTTOM] - bounds[TOP]
    current_ratio = width / height

    if current_ratio > aspect_ratio:
        # Too wide - shrink width, center horizontally
        new_width = int(height * aspect_ratio)
        delta = width - new_width
        bounds[LEFT] += delta // 2
        bounds[RIGHT] -= delta - delta // 2

    elif current_ratio < aspect_ratio:
        # Too tall - shrink height, center vertically
        new_height = int(width / aspect_ratio)
        delta = height - new_height
        bounds[TOP] += delta // 2
        bounds[BOTTOM] -= delta - delta // 2

    # Re-clamp after adjustment
    bounds[LEFT] = max(0, bounds[LEFT])
    bounds[RIGHT] = min(img_w, bounds[RIGHT])
    bounds[TOP] = max(y_min, bounds[TOP])
    bounds[BOTTOM] = min(y_max, bounds[BOTTOM])

    return bounds


def detect_frame_bounds(
    img: np.ndarray,
    aspect_ratio: float,
    visualizer: DebugVisualizer | None = None,
    crop_in_percent: float = 0.0,
) -> list[int]:
    """Detect frame boundaries in an image.

    Args:
        img: Input image as numpy array
        aspect_ratio: Expected aspect ratio of the frame (width/height)
        visualizer: Optional debug visualizer to save intermediate images
        crop_in_percent: Percentage to crop inward from edges (0-100)

    Returns:
        Array of [left, right, top, bottom] positions

    Raises:
        ValueError: If no lines or frame edges are detected, or invalid bounds
    """
    # Normalize levels to utilize full exposure range
    img = normalize_levels(img, visualizer)

    lines, y_min, y_max = detect_lines(img, visualizer)
    if lines is None:
        raise ValueError("No lines detected")

    horiz_positions, vert_positions = classify_lines(lines)

    if visualizer:
        visualizer.save_classified_lines(img, lines, horiz_positions, vert_positions)

    if not horiz_positions and not vert_positions:
        raise ValueError("No frame edges detected")

    img_h, img_w = img.shape[:2]

    # Get raw sizes to help estimate missing axis
    vert_span = (
        max(vert_positions) - min(vert_positions)
        if len(vert_positions) >= 2
        else img_w * 0.8
    )
    horiz_span = (
        max(horiz_positions) - min(horiz_positions)
        if len(horiz_positions) >= 2
        else img_h * 0.8
    )

    # Resolve edge positions
    left, right = resolve_axis(
        vert_positions, img_w, horiz_span, aspect_ratio, is_width=True
    )
    top, bottom = resolve_axis(
        horiz_positions, img_h, vert_span, aspect_ratio, is_width=False
    )

    # Build bounds array: [left, right, top, bottom]
    bounds = [left, right, top, bottom]

    # Clamp coordinates to image bounds (excluding sprocket regions)
    bounds[LEFT] = max(0, bounds[LEFT])
    bounds[RIGHT] = min(img_w, bounds[RIGHT])
    bounds[TOP] = max(y_min, bounds[TOP])
    bounds[BOTTOM] = min(y_max, bounds[BOTTOM])

    if bounds[RIGHT] <= bounds[LEFT] or bounds[BOTTOM] <= bounds[TOP]:
        raise ValueError("Invalid frame coordinates detected")

    # Apply crop-in to exclude unexposed borders
    bounds = apply_crop_in(img, bounds, crop_in_percent, visualizer)

    # Enforce correct aspect ratio
    bounds = enforce_aspect_ratio(bounds, aspect_ratio, y_min, y_max, img_w)

    if visualizer:
        visualizer.save_bounds(img, bounds)

    return bounds


def crop_frame(img: np.ndarray, bounds: list[int]) -> np.ndarray:
    """Crop image to the specified bounds.

    Args:
        img: Input image as numpy array
        bounds: Array of [left, right, top, bottom] positions

    Returns:
        Cropped image as numpy array
    """
    return img[bounds[TOP]:bounds[BOTTOM], bounds[LEFT]:bounds[RIGHT]]
