"""Frame detection and cropping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from models import EdgeGroup, FrameBounds, Line, Margins

if TYPE_CHECKING:
    from debug_visualizer import DebugVisualizer

# Re-export for backwards compatibility
parse_edge_margins = Margins.parse


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
) -> tuple[list[Line], int, int]:
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
    low_factor, high_factor = 0.66, 1.33
    low = int(max(0, low_factor * median))
    high = int(min(255, high_factor * median))
    edges = cv2.Canny(blurred, low, high)

    if visualizer:
        visualizer.save_edges(edges)
        visualizer.save_edges_variations(blurred, median, low_factor, high_factor)

    min_dim = min(img_cropped.shape[:2])
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_dim * 0.2,
        maxLineGap=min_dim // 10,
    )

    if raw_lines is None:
        return [], y_min, y_max

    # Convert to Line objects and adjust coordinates back to original image space
    lines = []
    for line_data in raw_lines[:, 0, :]:
        line = Line.from_hough(line_data)
        if y_offset_top > 0:
            line = line.offset_y(y_offset_top)
        lines.append(line)

    if visualizer:
        visualizer.save_lines(img, lines)

    return lines, y_min, y_max


def classify_lines(
    lines: list[Line],
    img_h: int,
    img_w: int,
    edge_margins: Margins,
) -> FrameBounds:
    """Classify lines into edge groups based on position and orientation.

    Only considers lines near the edges of the image to avoid false positives
    from image content in the middle.

    Args:
        lines: Detected Line objects
        img_h: Image height
        img_w: Image width
        edge_margins: Margins defining edge detection regions

    Returns:
        FrameBounds with lines grouped by edge
    """
    frame_bounds = FrameBounds()

    y_top_thresh = int(img_h * edge_margins.top)
    y_bottom_thresh = int(img_h * (1 - edge_margins.bottom))
    x_left_thresh = int(img_w * edge_margins.left)
    x_right_thresh = int(img_w * (1 - edge_margins.right))

    mid_y = img_h / 2
    mid_x = img_w / 2

    for line in lines:
        if line.is_horizontal:
            # Classify as top or bottom based on position
            if line.avg_y < y_top_thresh:
                frame_bounds.top.add(line)
            elif line.avg_y > y_bottom_thresh:
                frame_bounds.bottom.add(line)
        elif line.is_vertical:
            # Classify as left or right based on position
            if line.avg_x < x_left_thresh:
                frame_bounds.left.add(line)
            elif line.avg_x > x_right_thresh:
                frame_bounds.right.add(line)

    return frame_bounds


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
    left: int,
    right: int,
    top: int,
    bottom: int,
    crop_in_percent: float,
    visualizer: DebugVisualizer | None = None,
) -> tuple[int, int, int, int]:
    """Shrink bounds inward by a percentage to exclude unexposed borders.

    Args:
        img: Input image (used for visualization)
        left, right, top, bottom: Current bounds
        crop_in_percent: Percentage of each dimension to crop from edges (0-100)
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Adjusted bounds (left, right, top, bottom) with crop-in applied
    """
    if crop_in_percent <= 0:
        return left, right, top, bottom

    bounds_before = [left, right, top, bottom]

    width = right - left
    height = bottom - top

    crop_x = int(width * crop_in_percent / 100)
    crop_y = int(height * crop_in_percent / 100)

    left += crop_x
    right -= crop_x
    top += crop_y
    bottom -= crop_y

    if visualizer:
        bounds_after = [left, right, top, bottom]
        visualizer.save_crop_in(img, bounds_before, bounds_after, crop_in_percent)

    return left, right, top, bottom


def enforce_aspect_ratio(
    left: int,
    right: int,
    top: int,
    bottom: int,
    aspect_ratio: float,
    y_min: int,
    y_max: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Adjust bounds to enforce the correct aspect ratio.

    Shrinks the larger dimension to match the expected aspect ratio,
    keeping the crop centered.

    Args:
        left, right, top, bottom: Current bounds
        aspect_ratio: Expected aspect ratio (width/height)
        y_min: Minimum valid y coordinate (sprocket boundary)
        y_max: Maximum valid y coordinate (sprocket boundary)
        img_w: Image width for clamping

    Returns:
        Adjusted bounds (left, right, top, bottom) with correct aspect ratio
    """
    width = right - left
    height = bottom - top
    current_ratio = width / height

    if current_ratio > aspect_ratio:
        # Too wide - shrink width, center horizontally
        new_width = int(height * aspect_ratio)
        delta = width - new_width
        left += delta // 2
        right -= delta - delta // 2

    elif current_ratio < aspect_ratio:
        # Too tall - shrink height, center vertically
        new_height = int(width / aspect_ratio)
        delta = height - new_height
        top += delta // 2
        bottom -= delta - delta // 2

    # Re-clamp after adjustment
    left = max(0, left)
    right = min(img_w, right)
    top = max(y_min, top)
    bottom = min(y_max, bottom)

    return left, right, top, bottom


def detect_frame_bounds(
    img: np.ndarray,
    aspect_ratio: float,
    visualizer: DebugVisualizer | None = None,
    crop_in_percent: float = 0.0,
    edge_margins: Margins | None = None,
    ignore_margins: Margins | None = None,
) -> tuple[FrameBounds, list[int]]:
    """Detect frame boundaries in an image.

    Args:
        img: Input image as numpy array
        aspect_ratio: Expected aspect ratio of the frame (width/height)
        visualizer: Optional debug visualizer to save intermediate images
        crop_in_percent: Percentage to crop inward from edges (0-100)
        edge_margins: Margins defining edge detection regions
        ignore_margins: Margins to crop before analysis

    Returns:
        Tuple of (FrameBounds object, [left, right, top, bottom] positions)

    Raises:
        ValueError: If no lines or frame edges are detected, or invalid bounds
    """
    if edge_margins is None:
        edge_margins = Margins(0.3, 0.3, 0.3, 0.3)
    if ignore_margins is None:
        ignore_margins = Margins(0.0, 0.05, 0.0, 0.05)

    orig_h, orig_w = img.shape[:2]

    # Crop out ignored margins before analysis
    ignore_top = int(orig_h * ignore_margins.top)
    ignore_bottom = int(orig_h * ignore_margins.bottom)
    ignore_left = int(orig_w * ignore_margins.left)
    ignore_right = int(orig_w * ignore_margins.right)

    if visualizer:
        visualizer.save_ignore_margin(img, ignore_margins)

    if ignore_top > 0 or ignore_bottom > 0 or ignore_left > 0 or ignore_right > 0:
        img = img[
            ignore_top : orig_h - ignore_bottom,
            ignore_left : orig_w - ignore_right,
        ]

    # Normalize levels to utilize full exposure range
    img = normalize_levels(img, visualizer)

    lines, y_min, y_max = detect_lines(img, visualizer)
    if not lines:
        raise ValueError("No lines detected")

    img_h, img_w = img.shape[:2]

    if visualizer:
        visualizer.save_edge_margins(img, edge_margins)

    frame_bounds = classify_lines(lines, img_h, img_w, edge_margins)

    if visualizer:
        visualizer.save_classified_lines(img, frame_bounds)

    if not frame_bounds.top.lines and not frame_bounds.bottom.lines:
        if not frame_bounds.left.lines and not frame_bounds.right.lines:
            raise ValueError("No frame edges detected")

    # Get positions from edge groups
    top_positions = [line.avg_y for line in frame_bounds.top.lines]
    bottom_positions = [line.avg_y for line in frame_bounds.bottom.lines]
    left_positions = [line.avg_x for line in frame_bounds.left.lines]
    right_positions = [line.avg_x for line in frame_bounds.right.lines]

    # Get raw sizes to help estimate missing axis
    vert_span = (
        max(right_positions) - min(left_positions)
        if left_positions and right_positions
        else img_w * 0.8
    )
    horiz_span = (
        max(bottom_positions) - min(top_positions)
        if top_positions and bottom_positions
        else img_h * 0.8
    )

    # Resolve edge positions
    def resolve_axis(positions, img_size, other_size, ar, is_width):
        if len(positions) >= 2:
            return int(min(positions)), int(max(positions))
        if len(positions) == 1:
            edge_pos = int(positions[0])
            size = int(round(other_size * ar)) if is_width else int(round(other_size / ar))
            if edge_pos < img_size // 2:
                return edge_pos, edge_pos + size
            else:
                return edge_pos - size, edge_pos
        size = int(round(other_size * ar)) if is_width else int(round(other_size / ar))
        return (img_size - size) // 2, (img_size + size) // 2

    left, right = resolve_axis(
        left_positions + right_positions, img_w, horiz_span, aspect_ratio, is_width=True
    )
    top, bottom = resolve_axis(
        top_positions + bottom_positions, img_h, vert_span, aspect_ratio, is_width=False
    )

    # Clamp coordinates to image bounds (excluding sprocket regions)
    left = max(0, left)
    right = min(img_w, right)
    top = max(y_min, top)
    bottom = min(y_max, bottom)

    if right <= left or bottom <= top:
        raise ValueError("Invalid frame coordinates detected")

    # Apply crop-in to exclude unexposed borders
    left, right, top, bottom = apply_crop_in(
        img, left, right, top, bottom, crop_in_percent, visualizer
    )

    # Enforce correct aspect ratio
    left, right, top, bottom = enforce_aspect_ratio(
        left, right, top, bottom, aspect_ratio, y_min, y_max, img_w
    )

    if visualizer:
        visualizer.save_bounds(img, [left, right, top, bottom])

    # Adjust bounds back to original image coordinates
    if ignore_left > 0 or ignore_top > 0:
        left += ignore_left
        right += ignore_left
        top += ignore_top
        bottom += ignore_top

    return frame_bounds, [left, right, top, bottom]


def crop_frame(img: np.ndarray, bounds: list[int]) -> np.ndarray:
    """Crop image to the specified bounds.

    Args:
        img: Input image as numpy array
        bounds: Array of [left, right, top, bottom] positions

    Returns:
        Cropped image as numpy array
    """
    left, right, top, bottom = bounds
    return img[top:bottom, left:right]
