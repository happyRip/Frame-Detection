"""Frame detection and cropping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .models import EdgeGroup, FrameBounds, Line, Margins

if TYPE_CHECKING:
    from .visualizer import DebugVisualizer


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


def detect_film_base_color(
    img: np.ndarray,
    sprocket_mask: np.ndarray,
    y_min: int,
    y_max: int,
    visualizer: DebugVisualizer | None = None,
) -> np.ndarray:
    """Detect film base color from unexposed regions.

    Samples from sprocket regions (excluding holes) if detected,
    otherwise falls back to image edges.

    Args:
        img: Input image (normalized)
        sprocket_mask: Binary mask of sprocket holes (255=hole)
        y_min: Top boundary of valid frame region (below top sprockets)
        y_max: Bottom boundary of valid frame region (above bottom sprockets)
        visualizer: Optional debug visualizer

    Returns:
        Film base color as BGR numpy array of shape (3,)
    """
    img_h, img_w = img.shape[:2]
    sample_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    has_sprocket_regions = y_min > 0 or y_max < img_h

    if has_sprocket_regions:
        # Sample from sprocket regions, excluding the holes
        if y_min > 0:
            sample_mask[:y_min, :] = 255
        if y_max < img_h:
            sample_mask[y_max:, :] = 255
        # Exclude sprocket holes
        sample_mask[sprocket_mask > 0] = 0

    # Fallback: if no sprocket regions or insufficient samples, use image edges
    min_samples = 100
    if np.sum(sample_mask > 0) < min_samples:
        has_sprocket_regions = False
        sample_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        edge_size = max(10, int(min(img_h, img_w) * 0.05))
        sample_mask[:edge_size, :] = 255  # Top
        sample_mask[img_h - edge_size :, :] = 255  # Bottom
        sample_mask[:, :edge_size] = 255  # Left
        sample_mask[:, img_w - edge_size :] = 255  # Right

    # Extract sampled pixels and compute median color
    sampled_pixels = img[sample_mask > 0]
    film_base = np.median(sampled_pixels, axis=0).astype(np.uint8)

    if visualizer:
        visualizer.save_film_base(img, sample_mask, film_base, has_sprocket_regions)

    return film_base


def create_film_base_mask(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
    visualizer: DebugVisualizer | None = None,
) -> np.ndarray:
    """Create a mask of pixels that match the film base color.

    Args:
        img: Input image (normalized)
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching
        visualizer: Optional debug visualizer

    Returns:
        Binary mask where film base regions are 255
    """
    # Compute color distance from film base for each pixel
    diff = img.astype(np.float32) - film_base_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff**2, axis=2))

    # Create mask for pixels within tolerance
    film_base_mask = (distance <= tolerance).astype(np.uint8) * 255

    # Clean up with morphological operations
    kernel_size = max(3, int(min(img.shape[:2]) / 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    film_base_mask = cv2.morphologyEx(film_base_mask, cv2.MORPH_CLOSE, kernel)
    film_base_mask = cv2.morphologyEx(film_base_mask, cv2.MORPH_OPEN, kernel)

    if visualizer:
        visualizer.save_film_base_mask(img, film_base_mask, film_base_color, tolerance)

    return film_base_mask


def detect_lines(
    img: np.ndarray,
    film_base_mask: np.ndarray,
    edge_margins: Margins,
    visualizer: DebugVisualizer | None = None,
) -> list[Line]:
    """Detect lines from film base mask boundaries using Hough transform.

    Detects lines separately in each margin region (top, bottom, left, right)
    to ensure all frame edges are found.

    Args:
        img: Input image (for visualization only)
        film_base_mask: Binary mask where film base regions are 255
        edge_margins: Margins defining the regions to search for each edge
        visualizer: Optional debug visualizer

    Returns:
        List of detected Line objects
    """
    img_h, img_w = img.shape[:2]

    # Find edges of the film base mask (boundaries between frame and base)
    edges = cv2.Canny(film_base_mask, 50, 150)

    if visualizer:
        visualizer.save_edges(edges)

    # Define margin boundaries
    left_margin = int(img_w * edge_margins.left)
    right_margin = int(img_w * (1 - edge_margins.right))
    top_margin = int(img_h * edge_margins.top)
    bottom_margin = int(img_h * (1 - edge_margins.bottom))

    all_lines = []

    # Detect lines in each margin region separately
    regions = [
        ("left", edges[:, :left_margin], 0, 0),
        ("right", edges[:, right_margin:], 0, right_margin),
        ("top", edges[:top_margin, :], 0, 0),
        ("bottom", edges[bottom_margin:, :], bottom_margin, 0),
    ]

    for name, region_edges, y_offset, x_offset in regions:
        if region_edges.size == 0:
            continue

        # Calculate parameters based on region dimensions
        region_h, region_w = region_edges.shape[:2]
        min_dim = min(region_h, region_w)

        raw_lines = cv2.HoughLinesP(
            region_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(min_dim * 0.3),
            maxLineGap=min_dim // 5,
        )

        if raw_lines is not None:
            for line_data in raw_lines[:, 0, :]:
                # Offset coordinates back to full image space
                x1, y1, x2, y2 = line_data
                line_data_offset = [
                    x1 + x_offset,
                    y1 + y_offset,
                    x2 + x_offset,
                    y2 + y_offset,
                ]
                line = Line.from_hough(line_data_offset)
                all_lines.append(line)

    return all_lines


def classify_lines(
    lines: list[Line],
    img_h: int,
    img_w: int,
    edge_margins: Margins,
    y_min: int = 0,
    y_max: int | None = None,
) -> FrameBounds:
    """Classify lines into edge groups based on position and orientation.

    Only considers lines near the edges of the image to avoid false positives
    from image content in the middle.

    Args:
        lines: Detected Line objects
        img_h: Image height
        img_w: Image width
        edge_margins: Margins defining edge detection regions
        y_min: Minimum valid y coordinate (after sprocket cropping)
        y_max: Maximum valid y coordinate (after sprocket cropping)

    Returns:
        FrameBounds with lines grouped by edge
    """
    if y_max is None:
        y_max = img_h

    frame_bounds = FrameBounds()

    # Apply edge margins relative to the valid region (after sprocket cropping)
    valid_height = y_max - y_min
    y_top_thresh = y_min + int(valid_height * edge_margins.top)
    y_bottom_thresh = y_min + int(valid_height * (1 - edge_margins.bottom))
    x_left_thresh = int(img_w * edge_margins.left)
    x_right_thresh = int(img_w * (1 - edge_margins.right))

    mid_y = img_h / 2
    mid_x = img_w / 2

    for line in lines:
        if line.is_horizontal:
            # Classify as top or bottom only if within the appropriate margin
            if line.avg_y < y_top_thresh:
                frame_bounds.top.add(line)
            elif line.avg_y > y_bottom_thresh:
                frame_bounds.bottom.add(line)
            # Lines in the middle are ignored
        elif line.is_vertical:
            # Classify as left or right only if within the appropriate margin
            if line.avg_x < x_left_thresh:
                frame_bounds.left.add(line)
            elif line.avg_x > x_right_thresh:
                frame_bounds.right.add(line)
            # Lines in the middle are ignored

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
    film_base_mask: np.ndarray,
    left: int,
    right: int,
    top: int,
    bottom: int,
    max_crop_in_percent: float,
    visualizer: DebugVisualizer | None = None,
) -> tuple[int, int, int, int]:
    """Shrink bounds inward to exclude film base, up to a maximum percentage.

    Crops inward until no film base is visible, but won't exceed the maximum
    crop percentage on any side.

    Args:
        img: Input image (used for visualization)
        film_base_mask: Binary mask where film base regions are 255
        left, right, top, bottom: Current bounds
        max_crop_in_percent: Maximum percentage to crop from each edge (0-100)
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Adjusted bounds (left, right, top, bottom) with film base excluded
    """
    bounds_before = [left, right, top, bottom]

    width = right - left
    height = bottom - top

    # Calculate maximum allowed crop in pixels
    max_crop_x = int(width * max_crop_in_percent / 100)
    max_crop_y = int(height * max_crop_in_percent / 100)

    # Left: scan rightward until no film base, but don't exceed max
    new_left = left
    for x in range(left, min(left + max_crop_x, right)):
        col = film_base_mask[top:bottom, x]
        if not np.any(col > 0):
            new_left = x
            break
        new_left = x + 1
    left = min(new_left, left + max_crop_x)

    # Right: scan leftward until no film base, but don't exceed max
    new_right = right
    for x in range(right - 1, max(right - max_crop_x - 1, left), -1):
        col = film_base_mask[top:bottom, x]
        if not np.any(col > 0):
            new_right = x + 1
            break
        new_right = x
    right = max(new_right, right - max_crop_x)

    # Top: scan downward until no film base, but don't exceed max
    new_top = top
    for y in range(top, min(top + max_crop_y, bottom)):
        row = film_base_mask[y, left:right]
        if not np.any(row > 0):
            new_top = y
            break
        new_top = y + 1
    top = min(new_top, top + max_crop_y)

    # Bottom: scan upward until no film base, but don't exceed max
    new_bottom = bottom
    for y in range(bottom - 1, max(bottom - max_crop_y - 1, top), -1):
        row = film_base_mask[y, left:right]
        if not np.any(row > 0):
            new_bottom = y + 1
            break
        new_bottom = y
    bottom = max(new_bottom, bottom - max_crop_y)

    if visualizer:
        bounds_after = [left, right, top, bottom]
        visualizer.save_crop_in(img, bounds_before, bounds_after, max_crop_in_percent)

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

    img_h, img_w = img.shape[:2]

    # Step 1: Detect sprocket holes to find valid frame region
    sprocket_mask = detect_sprocket_holes(img, visualizer)
    _, y_offset_top, y_offset_bottom = crop_sprocket_region(
        img, sprocket_mask, visualizer
    )
    y_min = y_offset_top
    y_max = img_h - y_offset_bottom

    if y_min >= y_max:
        raise ValueError(
            f"Invalid frame region detected: y_min={y_min} >= y_max={y_max}. "
            "Sprocket detection may have failed for this image."
        )

    # Step 2: Normalize levels for better color detection
    img = normalize_levels(img, visualizer)

    # Step 3: Detect film base color from normalized image
    film_base_color = detect_film_base_color(
        img, sprocket_mask, y_min, y_max, visualizer
    )

    # Step 4: Create mask of film base regions
    film_base_mask = create_film_base_mask(img, film_base_color, visualizer=visualizer)

    # Crop mask to valid frame region to avoid detecting sprocket boundaries as edges
    film_base_mask_cropped = film_base_mask[y_min:y_max, :]

    if visualizer:
        visualizer.save_film_base_mask_cropped(img, film_base_mask_cropped, y_min, y_max)

    # Step 5: Detect lines from film base mask boundaries (in each margin region)
    img_cropped = img[y_min:y_max, :]
    lines = detect_lines(img_cropped, film_base_mask_cropped, edge_margins, visualizer)

    # Offset line coordinates back to original image space
    if y_min > 0:
        lines = [line.offset_y(y_min) for line in lines]

    if visualizer:
        visualizer.save_lines(img, lines)

    if not lines:
        raise ValueError("No lines detected")

    if visualizer:
        visualizer.save_edge_margins(img, edge_margins, y_min, y_max)

    frame_bounds = classify_lines(lines, img_h, img_w, edge_margins, y_min, y_max)

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

    # Determine which edges are detected
    has_left = len(left_positions) > 0
    has_right = len(right_positions) > 0
    has_top = len(top_positions) > 0
    has_bottom = len(bottom_positions) > 0

    # Get detected edge positions
    left = int(min(left_positions)) if has_left else None
    right = int(max(right_positions)) if has_right else None
    top = int(min(top_positions)) if has_top else None
    bottom = int(max(bottom_positions)) if has_bottom else None

    # Calculate missing edge using aspect ratio when we have 3 edges
    if has_top and has_bottom:
        height = bottom - top
        expected_width = int(height * aspect_ratio)
        if not has_left and has_right:
            left = right - expected_width
        elif not has_right and has_left:
            right = left + expected_width

    if has_left and has_right:
        width = right - left
        expected_height = int(width / aspect_ratio)
        if not has_top and has_bottom:
            top = bottom - expected_height
        elif not has_bottom and has_top:
            bottom = top + expected_height

    # Fallback for cases with fewer than 3 edges
    if left is None or right is None or top is None or bottom is None:
        # Estimate frame size from what we have
        if has_top and has_bottom:
            height = bottom - top
            expected_width = int(height * aspect_ratio)
        elif has_left and has_right:
            expected_width = right - left
        else:
            height = int((y_max - y_min) * 0.9)
            expected_width = int(height * aspect_ratio)

        if left is None and right is None:
            left = (img_w - expected_width) // 2
            right = (img_w + expected_width) // 2
        elif left is None:
            left = right - expected_width
        elif right is None:
            right = left + expected_width

        expected_height = int((right - left) / aspect_ratio)
        if top is None and bottom is None:
            center_y = (y_min + y_max) // 2
            top = center_y - expected_height // 2
            bottom = center_y + expected_height // 2
        elif top is None:
            top = bottom - expected_height
        elif bottom is None:
            bottom = top + expected_height

    # Clamp coordinates to image bounds (excluding sprocket regions)
    left = max(0, left)
    right = min(img_w, right)
    top = max(y_min, top)
    bottom = min(y_max, bottom)

    if right <= left or bottom <= top:
        raise ValueError("Invalid frame coordinates detected")

    # Apply crop-in to exclude film base from final crop
    left, right, top, bottom = apply_crop_in(
        img, film_base_mask, left, right, top, bottom, crop_in_percent, visualizer
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
