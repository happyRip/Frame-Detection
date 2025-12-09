"""Frame detection and cropping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .exceptions import (
    InvalidFrameCoordinatesError,
    InvalidFrameRegionError,
    NoFrameEdgesError,
    NoLinesDetectedError,
)
from .filters import EdgeFilter, apply_filter
from .models import (
    EdgeGroup,
    FilmCutEnd,
    FilmType,
    FilterConfig,
    FrameBounds,
    Line,
    Margins,
    Orientation,
)
from .separation import SeparationMethod, apply_separation

if TYPE_CHECKING:
    from .visualizer import DebugVisualizer


def _detect_bright_sprocket_holes(
    gray: np.ndarray, hist_smooth: np.ndarray
) -> tuple[np.ndarray, int | None]:
    """Detect sprocket holes as bright regions (for negative film).

    Args:
        gray: Grayscale image
        hist_smooth: Smoothed histogram

    Returns:
        Tuple of (sprocket_mask, threshold_idx)
    """
    sprocket_mask = np.zeros(gray.shape, dtype=np.uint8)
    threshold_idx = None

    # Find peaks in the bright region (right side of histogram, 200-255)
    bright_region = hist_smooth[200:]

    if bright_region.max() >= hist_smooth.max() * 0.02:
        # Find the rightmost significant peak
        # For sprocket holes, we want the rightmost peak (brightest values)
        # even if it's smaller than other peaks in the region

        # Find all local maxima (where value is greater than both neighbors)
        # Use low thresholds to catch smaller sprocket peaks
        min_prominence = bright_region.max() * 0.02  # 2% of max as prominence threshold
        min_height = bright_region.max() * 0.02  # Peak must be at least 2% of max height

        peaks = []
        for i in range(1, len(bright_region) - 1):
            if bright_region[i] > bright_region[i - 1] and bright_region[i] > bright_region[i + 1]:
                if bright_region[i] < min_height:
                    continue
                # Calculate prominence: height above the higher of the two surrounding valleys
                # Look left for valley
                left_min = bright_region[i]
                for j in range(i - 1, -1, -1):
                    if bright_region[j] < left_min:
                        left_min = bright_region[j]
                    if bright_region[j] > bright_region[i]:
                        break
                # Look right for valley
                right_min = bright_region[i]
                for j in range(i + 1, len(bright_region)):
                    if bright_region[j] < right_min:
                        right_min = bright_region[j]
                    if bright_region[j] > bright_region[i]:
                        break
                prominence = bright_region[i] - max(left_min, right_min)
                if prominence >= min_prominence:
                    peaks.append((i, bright_region[i], prominence))

        # Select the rightmost peak - this is most likely the sprocket holes
        peak_idx = None
        if peaks:
            # Sort by index descending (rightmost first) and take the first one
            peaks.sort(key=lambda x: x[0], reverse=True)
            peak_idx = 200 + peaks[0][0]

        # Fallback: if no clear peak found, find rightmost point above threshold
        if peak_idx is None:
            threshold = bright_region.max() * 0.3
            for i in range(len(bright_region) - 1, 0, -1):
                if bright_region[i] > threshold:
                    peak_idx = 200 + i
                    break

        if peak_idx is not None and peak_idx >= 210:
            # Find the valley (minimum) before the peak
            search_start = max(0, peak_idx - 200 - 30)  # Look up to 30 bins left of peak
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

    return sprocket_mask, threshold_idx


def _detect_dark_sprocket_holes(
    gray: np.ndarray, hist_smooth: np.ndarray
) -> tuple[np.ndarray, int | None]:
    """Detect sprocket holes as dark regions (for inverted negative film).

    For inverted negatives (negatives converted to positive), sprocket holes
    appear dark because the original clear areas become black when inverted.
    The film base appears medium-dark (inverted from orange/grey).

    This function analyzes only the edge regions where sprocket holes would be,
    and looks for pixels that are distinctly darker than the film base.

    Args:
        gray: Grayscale image
        hist_smooth: Smoothed histogram (of full image, used for reference)

    Returns:
        Tuple of (sprocket_mask, threshold_idx)
    """
    sprocket_mask = np.zeros(gray.shape, dtype=np.uint8)
    threshold_idx = None

    # Create edge mask to analyze only where sprocket holes would be
    edge_mask = _create_edge_mask(gray, edge_fraction=0.12)
    edge_pixels = gray[edge_mask > 0]

    if len(edge_pixels) == 0:
        return sprocket_mask, threshold_idx

    # Calculate histogram of edge pixels only
    edge_hist = cv2.calcHist([edge_pixels], [0], None, [256], [0, 256]).flatten()
    kernel_size = 5
    edge_hist_smooth = np.convolve(
        edge_hist, np.ones(kernel_size) / kernel_size, mode="same"
    )

    # Find percentiles of edge brightness
    edge_p10 = np.percentile(edge_pixels, 10)
    edge_p50 = np.percentile(edge_pixels, 50)  # median

    # For inverted negatives, sprocket holes are the darkest pixels at edges.
    # Use a more generous search range - up to median or 150, whichever is smaller
    search_limit = min(int(edge_p50), 150)

    if search_limit < 20:
        # Film base is already very dark, no room for darker sprocket holes
        return sprocket_mask, threshold_idx

    dark_region = edge_hist_smooth[:search_limit]

    if len(dark_region) == 0 or dark_region.max() == 0:
        return sprocket_mask, threshold_idx

    # Find if there's a peak in the dark region (relaxed threshold)
    # Accept peaks that are at least 0.5% of max histogram value
    if dark_region.max() < edge_hist_smooth.max() * 0.005:
        # No significant dark peak at edges
        return sprocket_mask, threshold_idx

    # Find the darkest significant peak
    threshold = dark_region.max() * 0.2
    peak_idx = None
    for i in range(len(dark_region)):
        if dark_region[i] > threshold:
            peak_idx = i
            break

    if peak_idx is None:
        # Fallback: use 10th percentile as threshold
        threshold_idx = int(edge_p10)
    else:
        # Find valley between dark peak and film base
        search_end = min(int(edge_p50) + 20, 200)
        if search_end <= peak_idx:
            search_end = peak_idx + 40

        region = edge_hist_smooth[peak_idx:search_end]
        if len(region) > 1:
            # Find the valley (minimum) after the peak
            valley_offset = np.argmin(region)
            valley_idx = peak_idx + valley_offset
            threshold_idx = valley_idx
        else:
            threshold_idx = peak_idx + 10

    # Ensure threshold is reasonable (between 10th percentile and median)
    threshold_idx = max(int(edge_p10), min(threshold_idx, int(edge_p50)))

    # Create mask - only consider edge pixels below threshold
    candidate_mask = (gray <= threshold_idx).astype(np.uint8) * 255

    # Only keep candidates that are at edges
    sprocket_mask = cv2.bitwise_and(candidate_mask, edge_mask)

    # Skip edge concentration check - the mask is already edge-constrained

    # Dilate to ensure we cover the hole edges
    dilate_k = max(3, int(min(gray.shape[:2]) / 150) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    sprocket_mask = cv2.dilate(sprocket_mask, kernel, iterations=2)

    return sprocket_mask, threshold_idx


def _check_edge_concentration(mask: np.ndarray, edge_fraction: float = 0.15) -> float:
    """Check what fraction of mask pixels are concentrated at image edges.

    Args:
        mask: Binary mask to analyze
        edge_fraction: Fraction of image dimension to consider as edge

    Returns:
        Ratio of edge pixels to total mask pixels (0-1)
    """
    if mask.max() == 0:
        return 0.0

    img_h, img_w = mask.shape[:2]
    h_margin = int(img_h * edge_fraction)
    w_margin = int(img_w * edge_fraction)

    total_pixels = np.sum(mask > 0)
    if total_pixels == 0:
        return 0.0

    # Count in edge regions
    top_bright = np.sum(mask[:h_margin, :] > 0)
    bottom_bright = np.sum(mask[img_h - h_margin:, :] > 0)
    left_bright = np.sum(mask[:, :w_margin] > 0)
    right_bright = np.sum(mask[:, img_w - w_margin:] > 0)

    edge_bright = top_bright + bottom_bright + left_bright + right_bright
    return min(1.0, edge_bright / total_pixels)


def _create_edge_mask(gray: np.ndarray, edge_fraction: float = 0.08) -> np.ndarray:
    """Create a mask for edge regions where sprocket holes would be located.

    Args:
        gray: Grayscale image
        edge_fraction: Fraction of image dimension to consider as edge (default 8%)

    Returns:
        Binary mask where edge regions are 255
    """
    img_h, img_w = gray.shape[:2]
    h_margin = int(img_h * edge_fraction)
    w_margin = int(img_w * edge_fraction)

    edge_mask = np.zeros(gray.shape, dtype=np.uint8)
    edge_mask[:h_margin, :] = 255  # Top edge
    edge_mask[img_h - h_margin:, :] = 255  # Bottom edge
    edge_mask[:, :w_margin] = 255  # Left edge
    edge_mask[:, img_w - w_margin:] = 255  # Right edge

    return edge_mask


def _analyze_edge_brightness(
    gray: np.ndarray, edge_fraction: float = 0.08
) -> tuple[float, float, np.ndarray]:
    """Analyze edge region brightness to detect film type.

    Samples the outer edges of the image where sprocket holes would be located,
    and analyzes brightness distribution (color-independent).

    Args:
        gray: Grayscale image
        edge_fraction: Fraction of image dimension to consider as edge (default 8%)

    Returns:
        Tuple of (bright_ratio, dark_ratio, edge_mask):
            - bright_ratio: Fraction of edge pixels that are bright (>200)
            - dark_ratio: Fraction of edge pixels that are dark (<50)
            - edge_mask: Binary mask showing the analyzed edge regions
    """
    edge_mask = _create_edge_mask(gray, edge_fraction)

    # Get pixels from edge regions
    edge_gray = gray[edge_mask > 0]
    total_edge_pixels = len(edge_gray)

    if total_edge_pixels == 0:
        return 0.0, 0.0, edge_mask

    # Count bright and dark pixels at edges
    bright_pixels = edge_gray[edge_gray >= 200]
    dark_pixels = edge_gray[edge_gray <= 50]

    bright_ratio = len(bright_pixels) / total_edge_pixels
    dark_ratio = len(dark_pixels) / total_edge_pixels

    return bright_ratio, dark_ratio, edge_mask


def _evaluate_sprocket_mask_quality(
    mask: np.ndarray, edge_fraction: float = 0.15
) -> tuple[float, float]:
    """Evaluate the quality of a sprocket mask.

    Args:
        mask: Binary sprocket mask
        edge_fraction: Fraction of image to consider as edge region

    Returns:
        Tuple of (coverage_ratio, edge_concentration):
            - coverage_ratio: Fraction of image covered by mask (0-1)
            - edge_concentration: Fraction of mask pixels at edges (0-1)
    """
    if mask.max() == 0:
        return 0.0, 0.0

    img_h, img_w = mask.shape[:2]
    total_pixels = img_h * img_w
    mask_pixels = np.sum(mask > 0)

    coverage_ratio = mask_pixels / total_pixels

    # Check edge concentration
    edge_concentration = _check_edge_concentration(mask, edge_fraction)

    return coverage_ratio, edge_concentration


def _auto_detect_film_type(
    gray: np.ndarray,
    hist_smooth: np.ndarray,
    img: np.ndarray | None = None,
    visualizer: DebugVisualizer | None = None,
) -> tuple[FilmType, np.ndarray, int | None, np.ndarray]:
    """Auto-detect film type based on sprocket mask quality comparison.

    Detects film type by running both bright and dark sprocket detection,
    then comparing which produces a better quality mask (valid coverage
    and edge concentration).

    For negative film (color or B&W):
        - Sprocket holes appear BRIGHT (light passes through clear holes)

    For positive/slide film or inverted negatives:
        - Sprocket holes appear DARK

    Args:
        gray: Grayscale image
        hist_smooth: Smoothed histogram
        img: Color image (for visualization)
        visualizer: Optional debug visualizer

    Returns:
        Tuple of (detected_film_type, sprocket_mask, threshold_idx, edge_analysis_mask)
    """
    # Try both detection methods
    bright_mask, bright_threshold = _detect_bright_sprocket_holes(gray, hist_smooth)
    dark_mask, dark_threshold = _detect_dark_sprocket_holes(gray, hist_smooth)

    # Get edge mask for visualization
    _, _, edge_mask = _analyze_edge_brightness(gray)

    # Evaluate quality of each mask
    bright_coverage, bright_edge_conc = _evaluate_sprocket_mask_quality(bright_mask)
    dark_coverage, dark_edge_conc = _evaluate_sprocket_mask_quality(dark_mask)

    # Valid sprocket mask criteria:
    # - Coverage between 0.5% and 25% of image
    # - Edge concentration > 50%
    min_coverage = 0.005
    max_coverage = 0.25
    min_edge_concentration = 0.5

    bright_valid = (
        min_coverage <= bright_coverage <= max_coverage
        and bright_edge_conc >= min_edge_concentration
    )
    dark_valid = (
        min_coverage <= dark_coverage <= max_coverage
        and dark_edge_conc >= min_edge_concentration
    )

    # Decision logic: prefer the mask that produces valid detection
    if dark_valid and not bright_valid:
        chosen, reason = "dark", "Dark valid, bright invalid"
        result = FilmType.POSITIVE, dark_mask, dark_threshold, edge_mask
    elif bright_valid and not dark_valid:
        chosen, reason = "bright", "Bright valid, dark invalid"
        result = FilmType.NEGATIVE, bright_mask, bright_threshold, edge_mask
    elif dark_valid and bright_valid:
        # Both valid - use edge concentration as tiebreaker
        if dark_edge_conc > bright_edge_conc:
            chosen, reason = "dark", f"Both valid, dark edge conc ({dark_edge_conc:.1%}) > bright ({bright_edge_conc:.1%})"
            result = FilmType.POSITIVE, dark_mask, dark_threshold, edge_mask
        else:
            chosen, reason = "bright", f"Both valid, bright edge conc ({bright_edge_conc:.1%}) >= dark ({dark_edge_conc:.1%})"
            result = FilmType.NEGATIVE, bright_mask, bright_threshold, edge_mask
    else:
        # Neither valid - fall back to brightness analysis
        bright_ratio, dark_ratio, _ = _analyze_edge_brightness(gray)
        if dark_ratio > bright_ratio:
            chosen, reason = "fallback-dark", f"Neither valid, fallback: dark_ratio ({dark_ratio:.1%}) > bright_ratio ({bright_ratio:.1%})"
            result = FilmType.POSITIVE, dark_mask, dark_threshold, edge_mask
        else:
            chosen, reason = "fallback-bright", f"Neither valid, fallback: bright_ratio ({bright_ratio:.1%}) >= dark_ratio ({dark_ratio:.1%})"
            result = FilmType.NEGATIVE, bright_mask, bright_threshold, edge_mask

    # Save comparison visualization
    if visualizer and img is not None:
        visualizer.save_sprocket_detection_comparison(
            img,
            bright_mask,
            dark_mask,
            bright_coverage,
            bright_edge_conc,
            dark_coverage,
            dark_edge_conc,
            chosen,
            reason,
        )

    return result


def detect_sprocket_holes(
    img: np.ndarray,
    film_type: FilmType = FilmType.AUTO,
    visualizer: DebugVisualizer | None = None,
) -> tuple[np.ndarray, FilmType]:
    """Detect sprocket holes based on film type.

    For negative film, sprocket holes appear as bright/white regions.
    For positive film, sprocket holes appear as dark/black regions.
    Auto mode detects based on histogram analysis and edge concentration.

    Args:
        img: Input image as numpy array
        film_type: Type of film (NEGATIVE, POSITIVE, or AUTO)
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Tuple of (binary mask where sprocket holes are marked as 255,
                  detected or specified film type)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Smooth histogram to reduce noise
    kernel_size = 5
    hist_smooth = np.convolve(hist, np.ones(kernel_size) / kernel_size, mode="same")

    # Compute edge brightness metrics and mask for visualization
    bright_ratio, dark_ratio, edge_mask = _analyze_edge_brightness(gray)
    edge_metrics = (bright_ratio, dark_ratio)

    if film_type == FilmType.AUTO:
        detected_type, sprocket_mask, threshold_idx, _ = _auto_detect_film_type(
            gray, hist_smooth, img, visualizer
        )
    elif film_type == FilmType.NEGATIVE:
        detected_type = FilmType.NEGATIVE
        sprocket_mask, threshold_idx = _detect_bright_sprocket_holes(gray, hist_smooth)
    else:  # FilmType.POSITIVE
        detected_type = FilmType.POSITIVE
        sprocket_mask, threshold_idx = _detect_dark_sprocket_holes(gray, hist_smooth)

    if visualizer:
        visualizer.save_sprocket_holes(
            img, sprocket_mask, hist_smooth, threshold_idx, detected_type,
            edge_metrics, edge_mask
        )

    return sprocket_mask, detected_type


def detect_sprocket_presence(
    sprocket_mask: np.ndarray,
    visualizer: DebugVisualizer | None = None,
) -> bool:
    """Detect if sprocket holes are present in the image.

    35mm film has sprocket holes on the long edges. Medium format and other
    films may not have sprocket holes at all. This function checks if the
    detected bright areas match the pattern of sprocket holes.

    Sprocket holes are present if:
    1. Significant bright areas are detected
    2. They are concentrated at edges (can be one or both edges)
    3. The density exceeds a minimum threshold

    Args:
        sprocket_mask: Binary mask where bright areas are marked as 255
        visualizer: Optional debug visualizer

    Returns:
        True if sprocket holes appear to be present, False otherwise
    """
    if sprocket_mask.max() == 0:
        if visualizer:
            visualizer.save_sprocket_presence(sprocket_mask, False, "No bright pixels", 0, 0)
        return False

    img_h, img_w = sprocket_mask.shape[:2]
    total_pixels = img_h * img_w

    # Check total bright area - sprocket holes should be a small but noticeable fraction
    total_bright = np.sum(sprocket_mask > 0)
    total_bright_ratio = total_bright / total_pixels

    # If too little bright area, no sprockets
    if total_bright_ratio < 0.005:  # Less than 0.5% of image
        if visualizer:
            visualizer.save_sprocket_presence(
                sprocket_mask, False, "Too few bright pixels", total_bright_ratio, 0
            )
        return False

    # If too much bright area, it's likely overexposure, not sprockets
    if total_bright_ratio > 0.25:  # More than 25% of image
        if visualizer:
            visualizer.save_sprocket_presence(
                sprocket_mask, False, "Too many bright pixels (overexposure?)",
                total_bright_ratio, 0
            )
        return False

    # Check if bright areas are concentrated at edges
    edge_fraction = 0.15
    h_margin = int(img_h * edge_fraction)
    w_margin = int(img_w * edge_fraction)

    # Count in edge regions
    top_bright = np.sum(sprocket_mask[:h_margin, :] > 0)
    bottom_bright = np.sum(sprocket_mask[img_h - h_margin:, :] > 0)
    left_bright = np.sum(sprocket_mask[:, :w_margin] > 0)
    right_bright = np.sum(sprocket_mask[:, img_w - w_margin:] > 0)

    # Total bright pixels at any edge
    edge_bright = top_bright + bottom_bright + left_bright + right_bright
    # Account for corner overlap
    edge_bright_ratio = min(1.0, edge_bright / total_bright) if total_bright > 0 else 0

    # For sprocket holes, most bright pixels should be at edges (at least 50%)
    has_sprockets = edge_bright_ratio > 0.5

    if visualizer:
        if has_sprockets:
            reason = "Sprockets detected at edges"
        else:
            reason = "Bright areas not concentrated at edges"
        visualizer.save_sprocket_presence(
            sprocket_mask, has_sprockets, reason, total_bright_ratio, edge_bright_ratio
        )

    return has_sprockets


def detect_sprocket_orientation(
    sprocket_mask: np.ndarray,
    visualizer: DebugVisualizer | None = None,
) -> Orientation:
    """Detect orientation of sprocket holes (top/bottom vs left/right).

    For 35mm film, sprocket holes run along both long edges. This function
    determines whether those edges are horizontal (top/bottom) or vertical
    (left/right) based on the distribution of detected sprocket pixels.

    Args:
        sprocket_mask: Binary mask where sprocket holes are marked as 255
        visualizer: Optional debug visualizer

    Returns:
        "horizontal" if sprockets are on top/bottom edges,
        "vertical" if sprockets are on left/right edges
    """
    if sprocket_mask.max() == 0:
        # No sprockets detected, default to horizontal
        return Orientation.HORIZONTAL

    img_h, img_w = sprocket_mask.shape[:2]

    # Define edge regions (outer 15% of each dimension)
    edge_fraction = 0.15
    h_margin = int(img_h * edge_fraction)
    w_margin = int(img_w * edge_fraction)

    # Count sprocket pixels in horizontal edges (top + bottom)
    top_region = sprocket_mask[:h_margin, :]
    bottom_region = sprocket_mask[img_h - h_margin :, :]
    horizontal_pixels = np.sum(top_region > 0) + np.sum(bottom_region > 0)

    # Count sprocket pixels in vertical edges (left + right)
    left_region = sprocket_mask[:, :w_margin]
    right_region = sprocket_mask[:, img_w - w_margin :]
    vertical_pixels = np.sum(left_region > 0) + np.sum(right_region > 0)

    # Determine orientation based on where sprocket pixels are concentrated
    # Use a ratio to account for different region sizes
    horizontal_density = horizontal_pixels / (2 * h_margin * img_w) if h_margin > 0 else 0
    vertical_density = vertical_pixels / (2 * w_margin * img_h) if w_margin > 0 else 0

    orientation = (
        Orientation.VERTICAL if vertical_density > horizontal_density else Orientation.HORIZONTAL
    )

    if visualizer:
        visualizer.save_sprocket_orientation(
            sprocket_mask, orientation, horizontal_density, vertical_density
        )

    return orientation


def detect_film_cut_end(
    sprocket_mask: np.ndarray,
    orientation: Orientation,
    visualizer: DebugVisualizer | None = None,
) -> FilmCutEnd:
    """Detect if the film has been cut and the cut end is visible.

    When a film strip is cut, the cut end may be visible in the viewport as a
    large bright area at the edge (no film covering it). This can only happen
    at the ends of the film strip - in landscape orientation, this is left/right.

    The detection looks for bright areas at the edge that:
    1. Touch the viewport edge
    2. Span a significant portion of the frame height
    3. Are contiguous (not isolated sprocket holes)

    Args:
        sprocket_mask: Binary mask where bright areas (sprocket holes) are 255
        orientation: Film orientation
        visualizer: Optional debug visualizer

    Returns:
        FilmCutEnd object indicating which edges have a visible cut end
    """
    if sprocket_mask.max() == 0:
        return FilmCutEnd()

    img_h, img_w = sprocket_mask.shape[:2]
    cut_end = FilmCutEnd()

    # Define edge region width and minimum coverage threshold
    edge_width_fraction = 0.05  # Check outer 5% of each edge
    min_coverage_fraction = 0.3  # Must cover at least 30% of the edge length

    if orientation == Orientation.HORIZONTAL:
        # Film runs left-right, so cut ends appear on left/right edges
        edge_width = max(10, int(img_w * edge_width_fraction))

        # Check left edge
        left_region = sprocket_mask[:, :edge_width]
        left_cols_with_bright = np.any(left_region > 0, axis=1)
        left_coverage = np.sum(left_cols_with_bright) / img_h
        if left_coverage >= min_coverage_fraction:
            # Check if it's contiguous (not scattered sprocket holes)
            # Find the largest contiguous vertical span
            in_span = False
            max_span = 0
            current_span = 0
            for has_bright in left_cols_with_bright:
                if has_bright:
                    if not in_span:
                        in_span = True
                        current_span = 1
                    else:
                        current_span += 1
                else:
                    if in_span:
                        max_span = max(max_span, current_span)
                        in_span = False
            if in_span:
                max_span = max(max_span, current_span)

            # If the largest span is significant, it's likely a cut end
            if max_span / img_h >= min_coverage_fraction:
                cut_end.left = True

        # Check right edge
        right_region = sprocket_mask[:, img_w - edge_width :]
        right_cols_with_bright = np.any(right_region > 0, axis=1)
        right_coverage = np.sum(right_cols_with_bright) / img_h
        if right_coverage >= min_coverage_fraction:
            in_span = False
            max_span = 0
            current_span = 0
            for has_bright in right_cols_with_bright:
                if has_bright:
                    if not in_span:
                        in_span = True
                        current_span = 1
                    else:
                        current_span += 1
                else:
                    if in_span:
                        max_span = max(max_span, current_span)
                        in_span = False
            if in_span:
                max_span = max(max_span, current_span)

            if max_span / img_h >= min_coverage_fraction:
                cut_end.right = True

    else:
        # Film runs top-bottom, so cut ends appear on top/bottom edges
        edge_height = max(10, int(img_h * edge_width_fraction))

        # Check top edge
        top_region = sprocket_mask[:edge_height, :]
        top_rows_with_bright = np.any(top_region > 0, axis=0)
        top_coverage = np.sum(top_rows_with_bright) / img_w
        if top_coverage >= min_coverage_fraction:
            in_span = False
            max_span = 0
            current_span = 0
            for has_bright in top_rows_with_bright:
                if has_bright:
                    if not in_span:
                        in_span = True
                        current_span = 1
                    else:
                        current_span += 1
                else:
                    if in_span:
                        max_span = max(max_span, current_span)
                        in_span = False
            if in_span:
                max_span = max(max_span, current_span)

            if max_span / img_w >= min_coverage_fraction:
                cut_end.top = True

        # Check bottom edge
        bottom_region = sprocket_mask[img_h - edge_height :, :]
        bottom_rows_with_bright = np.any(bottom_region > 0, axis=0)
        bottom_coverage = np.sum(bottom_rows_with_bright) / img_w
        if bottom_coverage >= min_coverage_fraction:
            in_span = False
            max_span = 0
            current_span = 0
            for has_bright in bottom_rows_with_bright:
                if has_bright:
                    if not in_span:
                        in_span = True
                        current_span = 1
                    else:
                        current_span += 1
                else:
                    if in_span:
                        max_span = max(max_span, current_span)
                        in_span = False
            if in_span:
                max_span = max(max_span, current_span)

            if max_span / img_w >= min_coverage_fraction:
                cut_end.bottom = True

    if visualizer:
        visualizer.save_film_cut_end(sprocket_mask, orientation, cut_end)

    return cut_end


def filter_cut_end_from_sprocket_mask(
    sprocket_mask: np.ndarray,
    orientation: Orientation,
    cut_end: FilmCutEnd,
) -> np.ndarray:
    """Remove cut end regions from the sprocket mask.

    This prevents the cut end from being detected as sprocket holes and
    affecting sprocket cropping.

    Args:
        sprocket_mask: Binary mask where sprocket holes are 255
        orientation: Film orientation
        cut_end: Detected film cut ends

    Returns:
        Filtered sprocket mask with cut end regions removed
    """
    if not cut_end.any_detected:
        return sprocket_mask

    filtered_mask = sprocket_mask.copy()
    img_h, img_w = sprocket_mask.shape[:2]

    # Remove a larger region to ensure the cut end doesn't affect cropping
    removal_fraction = 0.10  # Remove outer 10% where cut is detected

    if orientation == Orientation.HORIZONTAL:
        removal_width = max(10, int(img_w * removal_fraction))
        if cut_end.left:
            filtered_mask[:, :removal_width] = 0
        if cut_end.right:
            filtered_mask[:, img_w - removal_width :] = 0
    else:
        removal_height = max(10, int(img_h * removal_fraction))
        if cut_end.top:
            filtered_mask[:removal_height, :] = 0
        if cut_end.bottom:
            filtered_mask[img_h - removal_height :, :] = 0

    return filtered_mask


def _fit_sprocket_boundary(
    sprocket_mask: np.ndarray,
    axis: int,
    half: str,
    poly_degree: int = 2,
) -> np.ndarray | None:
    """Fit a polynomial curve to the sprocket boundary.

    For each position along the axis, finds the innermost sprocket pixel
    and fits a polynomial to these boundary points.

    Args:
        sprocket_mask: Binary mask where sprocket holes are 255
        axis: 0 for horizontal boundary (top/bottom), 1 for vertical (left/right)
        half: "top", "bottom", "left", or "right" - which half to analyze
        poly_degree: Degree of polynomial to fit (default 2 for quadratic)

    Returns:
        Array of fitted boundary positions for each position along the edge,
        or None if no sprockets found in this half.
    """
    img_h, img_w = sprocket_mask.shape[:2]

    if axis == 0:  # Horizontal boundary (top/bottom sprockets)
        mid = img_h // 2
        positions = []
        x_coords = []

        for x in range(img_w):
            col = sprocket_mask[:, x]
            sprocket_rows = np.where(col > 0)[0]

            if half == "top":
                top_sprockets = sprocket_rows[sprocket_rows < mid]
                if len(top_sprockets) > 0:
                    # Innermost = maximum y for top sprockets
                    positions.append(np.max(top_sprockets))
                    x_coords.append(x)
            else:  # bottom
                bottom_sprockets = sprocket_rows[sprocket_rows >= mid]
                if len(bottom_sprockets) > 0:
                    # Innermost = minimum y for bottom sprockets
                    positions.append(np.min(bottom_sprockets))
                    x_coords.append(x)

        if len(positions) < poly_degree + 1:
            return None

        # Fit polynomial
        x_coords = np.array(x_coords)
        positions = np.array(positions)
        coeffs = np.polyfit(x_coords, positions, poly_degree)
        fitted = np.polyval(coeffs, np.arange(img_w))

        return fitted

    else:  # Vertical boundary (left/right sprockets)
        mid = img_w // 2
        positions = []
        y_coords = []

        for y in range(img_h):
            row = sprocket_mask[y, :]
            sprocket_cols = np.where(row > 0)[0]

            if half == "left":
                left_sprockets = sprocket_cols[sprocket_cols < mid]
                if len(left_sprockets) > 0:
                    # Innermost = maximum x for left sprockets
                    positions.append(np.max(left_sprockets))
                    y_coords.append(y)
            else:  # right
                right_sprockets = sprocket_cols[sprocket_cols >= mid]
                if len(right_sprockets) > 0:
                    # Innermost = minimum x for right sprockets
                    positions.append(np.min(right_sprockets))
                    y_coords.append(y)

        if len(positions) < poly_degree + 1:
            return None

        # Fit polynomial
        y_coords = np.array(y_coords)
        positions = np.array(positions)
        coeffs = np.polyfit(y_coords, positions, poly_degree)
        fitted = np.polyval(coeffs, np.arange(img_h))

        return fitted


def crop_sprocket_region(
    img: np.ndarray,
    sprocket_mask: np.ndarray,
    orientation: Orientation = Orientation.HORIZONTAL,
    margin_percent: float = 0.0,
    visualizer: DebugVisualizer | None = None,
) -> tuple[np.ndarray, int, int, int, int, np.ndarray | None, np.ndarray | None]:
    """Crop out the region containing sprocket holes.

    Sprocket holes on 35mm film run along both long edges. Depending on
    orientation, this crops from top/bottom (horizontal) or left/right (vertical).

    Uses polynomial curve fitting to account for misaligned sprocket holes,
    then crops at the most conservative (innermost) point of the fitted curve.

    Args:
        img: Input image as numpy array
        sprocket_mask: Binary mask where sprocket holes are marked as 255
        orientation: "horizontal" for top/bottom sprockets, "vertical" for left/right
        margin_percent: Additional percentage of image dimension to crop beyond
            detected sprocket boundary (default 0.0, e.g., 0.5 for 0.5%)
        visualizer: Optional debug visualizer to save intermediate images

    Returns:
        Tuple of (cropped_image, y_offset_top, y_offset_bottom, x_offset_left, x_offset_right,
        curve1, curve2) where offsets indicate pixels cropped from each edge and curves
        are the fitted polynomial boundaries.
    """
    if sprocket_mask.max() == 0:
        if visualizer:
            visualizer.save_sprocket_crop(img, img, 0, 0, orientation)
        return img, 0, 0, 0, 0, None, None

    img_h, img_w = img.shape[:2]

    if orientation == Orientation.HORIZONTAL:
        # Sprockets on top/bottom - crop rows
        mid_y = img_h // 2
        margin = int(img_h * margin_percent / 100)

        # Fit curves to top and bottom boundaries
        top_curve = _fit_sprocket_boundary(sprocket_mask, axis=0, half="top")
        bottom_curve = _fit_sprocket_boundary(sprocket_mask, axis=0, half="bottom")

        # Top crop: use maximum of fitted curve (most conservative)
        if top_curve is not None:
            crop_top = int(np.max(top_curve)) + 1 + margin
            crop_top = min(crop_top, mid_y)
        else:
            crop_top = 0

        # Bottom crop: use minimum of fitted curve (most conservative)
        if bottom_curve is not None:
            crop_bottom = int(np.min(bottom_curve)) - margin
            crop_bottom = max(crop_bottom, mid_y)
        else:
            crop_bottom = img_h

        img_cropped = img[crop_top:crop_bottom, :]

        if visualizer:
            visualizer.save_sprocket_crop(
                img, img_cropped, crop_top, img_h - crop_bottom, orientation,
                top_curve, bottom_curve
            )

        return img_cropped, crop_top, img_h - crop_bottom, 0, 0, top_curve, bottom_curve

    else:
        # Sprockets on left/right - crop columns
        mid_x = img_w // 2
        margin = int(img_w * margin_percent / 100)

        # Fit curves to left and right boundaries
        left_curve = _fit_sprocket_boundary(sprocket_mask, axis=1, half="left")
        right_curve = _fit_sprocket_boundary(sprocket_mask, axis=1, half="right")

        # Left crop: use maximum of fitted curve (most conservative)
        if left_curve is not None:
            crop_left = int(np.max(left_curve)) + 1 + margin
            crop_left = min(crop_left, mid_x)
        else:
            crop_left = 0

        # Right crop: use minimum of fitted curve (most conservative)
        if right_curve is not None:
            crop_right = int(np.min(right_curve)) - margin
            crop_right = max(crop_right, mid_x)
        else:
            crop_right = img_w

        img_cropped = img[:, crop_left:crop_right]

        if visualizer:
            visualizer.save_sprocket_crop(
                img, img_cropped, crop_left, img_w - crop_right, orientation,
                left_curve, right_curve
            )

        return img_cropped, 0, 0, crop_left, img_w - crop_right, left_curve, right_curve


def mask_sprocket_region(
    mask: np.ndarray,
    orientation: Orientation,
    curve1: np.ndarray | None,
    curve2: np.ndarray | None,
    margin: int = 1,
) -> np.ndarray:
    """Zero out sprocket regions in a mask using fitted curves.

    Args:
        mask: Binary mask to modify
        orientation: Film orientation
        curve1: Fitted curve for top (horizontal) or left (vertical) boundary
        curve2: Fitted curve for bottom (horizontal) or right (vertical) boundary
        margin: Extra pixels to mask inward from the curve boundary

    Returns:
        Mask with sprocket regions zeroed out
    """
    result = mask.copy()
    img_h, img_w = mask.shape[:2]

    if orientation == Orientation.HORIZONTAL:
        for x in range(img_w):
            if curve1 is not None and x < len(curve1):
                y_top = max(0, int(curve1[x]) + margin)
                result[:y_top, x] = 0
            if curve2 is not None and x < len(curve2):
                y_bottom = min(img_h, int(curve2[x]) - margin)
                result[y_bottom:, x] = 0
    else:
        for y in range(img_h):
            if curve1 is not None and y < len(curve1):
                x_left = max(0, int(curve1[y]) + margin)
                result[y, :x_left] = 0
            if curve2 is not None and y < len(curve2):
                x_right = min(img_w, int(curve2[y]) - margin)
                result[y, x_right:] = 0

    return result


def mask_cut_end_region(
    mask: np.ndarray,
    orientation: Orientation,
    cut_end: FilmCutEnd,
    removal_fraction: float = 0.10,
) -> np.ndarray:
    """Zero out cut end regions in a mask.

    Args:
        mask: Binary mask to modify
        orientation: Film orientation
        cut_end: Detected cut end edges
        removal_fraction: Fraction of image to remove at cut ends

    Returns:
        Mask with cut end regions zeroed out
    """
    result = mask.copy()
    img_h, img_w = mask.shape[:2]

    if orientation == Orientation.HORIZONTAL:
        removal_width = max(10, int(img_w * removal_fraction))
        if cut_end.left:
            result[:, :removal_width] = 0
        if cut_end.right:
            result[:, img_w - removal_width:] = 0
    else:
        removal_height = max(10, int(img_h * removal_fraction))
        if cut_end.top:
            result[:removal_height, :] = 0
        if cut_end.bottom:
            result[img_h - removal_height:, :] = 0

    return result


def detect_film_base_color(
    img: np.ndarray,
    sprocket_mask: np.ndarray,
    orientation: Orientation = Orientation.HORIZONTAL,
    curve1: np.ndarray | None = None,
    curve2: np.ndarray | None = None,
    aspect_ratio: float | None = None,
    inset_percent: float = 1.0,
    visualizer: DebugVisualizer | None = None,
) -> np.ndarray:
    """Detect film base color from unexposed regions.

    When sprocket curves are available, samples from sprocket regions
    (excluding holes). Otherwise, samples from outside a centered rectangle
    with the specified aspect ratio, inset by a percentage of the diagonal.

    Args:
        img: Input image (normalized)
        sprocket_mask: Binary mask of sprocket holes (255=hole)
        orientation: "horizontal" for top/bottom sprockets, "vertical" for left/right
        curve1: Fitted curve for top (horizontal) or left (vertical) boundary
        curve2: Fitted curve for bottom (horizontal) or right (vertical) boundary
        aspect_ratio: Aspect ratio (width/height) for inner rectangle (no sprockets)
        inset_percent: Diagonal inset percentage for inner rectangle (0-50)
        visualizer: Optional debug visualizer

    Returns:
        Film base color as BGR numpy array of shape (3,)
    """
    img_h, img_w = img.shape[:2]

    sample_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    has_sprocket_regions = curve1 is not None or curve2 is not None
    inner_rect = None

    if has_sprocket_regions:
        if orientation == Orientation.HORIZONTAL:
            # Sample from regions above curve1 (top) and below curve2 (bottom)
            for x in range(img_w):
                if curve1 is not None:
                    y_top = int(curve1[x])
                    if y_top > 0:
                        sample_mask[:y_top, x] = 255
                if curve2 is not None:
                    y_bottom = int(curve2[x])
                    if y_bottom < img_h:
                        sample_mask[y_bottom:, x] = 255
        else:
            # Sample from regions left of curve1 and right of curve2
            for y in range(img_h):
                if curve1 is not None:
                    x_left = int(curve1[y])
                    if x_left > 0:
                        sample_mask[y, :x_left] = 255
                if curve2 is not None:
                    x_right = int(curve2[y])
                    if x_right < img_w:
                        sample_mask[y, x_right:] = 255

        # Exclude sprocket holes
        sample_mask[sprocket_mask > 0] = 0

    else:
        # No sprocket regions - use aspect ratio based sampling
        if aspect_ratio is not None:
            # The inner rectangle has the film aspect ratio.
            # The tight axis (less room) uses inset_percent as margin.
            # The loose axis margin is calculated to maintain the aspect ratio.
            img_ar = img_w / img_h

            if img_ar > aspect_ratio:
                # Image is wider than film - vertical axis is tight (height limiting)
                margin_y = img_h * inset_percent / 100
                inner_h = img_h - 2 * margin_y
                inner_w = inner_h * aspect_ratio
                margin_x = (img_w - inner_w) / 2
            else:
                # Image is taller than film - horizontal axis is tight (width limiting)
                margin_x = img_w * inset_percent / 100
                inner_w = img_w - 2 * margin_x
                inner_h = inner_w / aspect_ratio
                margin_y = (img_h - inner_h) / 2

            inner_left = int(margin_x)
            inner_right = int(img_w - margin_x)
            inner_top = int(margin_y)
            inner_bottom = int(img_h - margin_y)

            # Sample from outside inner rectangle
            sample_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
            if inner_right > inner_left and inner_bottom > inner_top:
                sample_mask[inner_top:inner_bottom, inner_left:inner_right] = 0

            inner_rect = (inner_left, inner_top, inner_right, inner_bottom)

            # Exclude sprocket holes (just in case)
            sample_mask[sprocket_mask > 0] = 0

    # Fallback: if insufficient samples, use image edges
    min_samples = 100
    if np.sum(sample_mask > 0) < min_samples:
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
        visualizer.save_film_base(
            img, sample_mask, film_base, has_sprocket_regions, inset_percent, inner_rect
        )

    return film_base


def create_film_base_mask(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
    visualizer: DebugVisualizer | None = None,
    separation_method: SeparationMethod = SeparationMethod.COLOR_DISTANCE,
    separation_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Create a mask of pixels that match the film base color.

    Args:
        img: Input image (normalized)
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching
        visualizer: Optional debug visualizer
        separation_method: Method to use for separating film base from image
        separation_params: Optional dict of method-specific parameters

    Returns:
        Binary mask where film base regions are 255
    """
    # Apply selected separation method
    film_base_mask = apply_separation(
        img, film_base_color, separation_method, tolerance, separation_params
    )

    # Clean up with morphological operations
    kernel_size = max(3, int(min(img.shape[:2]) / 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    film_base_mask = cv2.morphologyEx(film_base_mask, cv2.MORPH_CLOSE, kernel)
    film_base_mask = cv2.morphologyEx(film_base_mask, cv2.MORPH_OPEN, kernel)

    if visualizer:
        visualizer.save_film_base_mask(img, film_base_mask, film_base_color, tolerance)

    return film_base_mask


def _calculate_ratio_aware_margins(
    img_w: int,
    img_h: int,
    aspect_ratio: float,
    edge_margin_percent: float,
    y_min: int = 0,
    y_max: int | None = None,
    x_min: int = 0,
    x_max: int | None = None,
) -> tuple[int, int, int, int]:
    """Calculate edge margins so the inner zone has the correct aspect ratio.

    Uses the tight/loose approach similar to film base detection:
    - The tight axis (less room relative to target ratio) uses edge_margin_percent
    - The loose axis margin is calculated to maintain the aspect ratio

    Args:
        img_w: Image width
        img_h: Image height
        aspect_ratio: Target aspect ratio (width/height) for the inner zone
        edge_margin_percent: Margin percentage for the tight axis (0-1, e.g., 0.3 for 30%)
        y_min: Minimum valid y coordinate (after sprocket cropping)
        y_max: Maximum valid y coordinate (after sprocket cropping)
        x_min: Minimum valid x coordinate (after sprocket cropping)
        x_max: Maximum valid x coordinate (after sprocket cropping)

    Returns:
        Tuple of (left_margin, right_margin, top_margin, bottom_margin) as pixel positions
        where left_margin/top_margin are from edge, right_margin/bottom_margin are from edge
    """
    if y_max is None:
        y_max = img_h
    if x_max is None:
        x_max = img_w

    # Work within the valid region (after sprocket cropping)
    valid_width = x_max - x_min
    valid_height = y_max - y_min

    if valid_width <= 0 or valid_height <= 0:
        # Fallback: use symmetric margins
        return (
            int(img_w * edge_margin_percent),
            int(img_w * (1 - edge_margin_percent)),
            int(img_h * edge_margin_percent),
            int(img_h * (1 - edge_margin_percent)),
        )

    # Calculate the aspect ratio of the valid region
    valid_ar = valid_width / valid_height

    if valid_ar > aspect_ratio:
        # Valid region is wider than target - vertical axis is tight (height limiting)
        # Use edge_margin_percent for top/bottom margins
        margin_y = valid_height * edge_margin_percent
        inner_h = valid_height - 2 * margin_y
        inner_w = inner_h * aspect_ratio
        margin_x = (valid_width - inner_w) / 2
    else:
        # Valid region is taller than target - horizontal axis is tight (width limiting)
        # Use edge_margin_percent for left/right margins
        margin_x = valid_width * edge_margin_percent
        inner_w = valid_width - 2 * margin_x
        inner_h = inner_w / aspect_ratio
        margin_y = (valid_height - inner_h) / 2

    # Convert to absolute pixel positions
    left_margin = x_min + int(margin_x)
    right_margin = x_min + int(valid_width - margin_x)
    top_margin = y_min + int(margin_y)
    bottom_margin = y_min + int(valid_height - margin_y)

    # Ensure margins are within bounds
    left_margin = max(0, min(left_margin, img_w))
    right_margin = max(0, min(right_margin, img_w))
    top_margin = max(0, min(top_margin, img_h))
    bottom_margin = max(0, min(bottom_margin, img_h))

    return left_margin, right_margin, top_margin, bottom_margin


def detect_lines(
    img: np.ndarray,
    film_base_mask: np.ndarray,
    edge_margins: Margins,
    orientation: Orientation = Orientation.HORIZONTAL,
    curve1: np.ndarray | None = None,
    curve2: np.ndarray | None = None,
    aspect_ratio: float | None = None,
    y_min: int = 0,
    y_max: int | None = None,
    x_min: int = 0,
    x_max: int | None = None,
    visualizer: DebugVisualizer | None = None,
    edge_filter: EdgeFilter = EdgeFilter.CANNY,
    cut_end: FilmCutEnd | None = None,
    sprocket_margin_percent: float = 0.0,
    edge_filter_params: dict[str, Any] | None = None,
) -> tuple[list[Line], np.ndarray]:
    """Detect lines from film base mask boundaries using Hough transform.

    Detects lines separately in each margin region (top, bottom, left, right)
    to ensure all frame edges are found.

    When aspect_ratio is provided, uses the tight/loose approach to calculate
    margins so that the inner zone (where no lines are searched) has the
    correct aspect ratio. The tight axis uses the edge_margin percentage,
    and the loose axis margin is calculated to maintain the ratio.

    Args:
        img: Input image (for visualization only)
        film_base_mask: Binary mask where film base regions are 255
        edge_margins: Margins defining the regions to search for each edge
            (uses the maximum of left/right or top/bottom as the tight margin)
        orientation: Film orientation for curve masking
        curve1: Fitted curve for top (horizontal) or left (vertical) boundary
        curve2: Fitted curve for bottom (horizontal) or right (vertical) boundary
        aspect_ratio: Target aspect ratio for ratio-aware margin calculation
        y_min: Minimum valid y coordinate (after sprocket cropping)
        y_max: Maximum valid y coordinate (after sprocket cropping)
        x_min: Minimum valid x coordinate (after sprocket cropping)
        x_max: Maximum valid x coordinate (after sprocket cropping)
        visualizer: Optional debug visualizer
        edge_filter: Edge detection filter method to use
        cut_end: Detected film cut ends to mask from edges
        sprocket_margin_percent: Additional percentage to mask beyond sprocket curves
        edge_filter_params: Optional dict of filter-specific parameters

    Returns:
        Tuple of (list of detected Line objects, edges image)
    """
    img_h, img_w = img.shape[:2]

    # Find edges of the film base mask (boundaries between frame and base)
    edges = apply_filter(film_base_mask, edge_filter, edge_filter_params)

    # Mask out sprocket regions from edges using curves
    if curve1 is not None or curve2 is not None:
        # Calculate sprocket margin in pixels based on orientation
        if orientation == Orientation.HORIZONTAL:
            sprocket_margin_px = max(1, int(img_h * sprocket_margin_percent / 100))
        else:
            sprocket_margin_px = max(1, int(img_w * sprocket_margin_percent / 100))
        edges = mask_sprocket_region(edges, orientation, curve1, curve2, sprocket_margin_px)

    # Mask out cut end regions from edges
    if cut_end is not None and cut_end.any_detected:
        edges = mask_cut_end_region(edges, orientation, cut_end)

    # Define margin boundaries
    if aspect_ratio is not None:
        # Use ratio-aware margins: tight axis uses the specified margin,
        # loose axis is calculated to maintain aspect ratio
        edge_margin_percent = max(
            edge_margins.left, edge_margins.right,
            edge_margins.top, edge_margins.bottom
        )
        left_margin, right_margin, top_margin, bottom_margin = _calculate_ratio_aware_margins(
            img_w, img_h, aspect_ratio, edge_margin_percent,
            y_min, y_max, x_min, x_max
        )
    else:
        # Use fixed margins as before
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

    # Calculate Hough parameters based on the valid region size, not search region size
    # This ensures consistent line detection regardless of margin calculation method
    if y_max is None:
        y_max = img_h
    if x_max is None:
        x_max = img_w
    valid_height = y_max - y_min
    valid_width = x_max - x_min
    base_dim = min(valid_height, valid_width) if valid_height > 0 and valid_width > 0 else min(img_h, img_w)

    for name, region_edges, y_offset, x_offset in regions:
        if region_edges.size == 0:
            continue

        # Use base_dim (from valid region) for consistent Hough parameters
        raw_lines = cv2.HoughLinesP(
            region_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(base_dim * 0.3),
            maxLineGap=base_dim // 5,
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

    return all_lines, edges


def classify_lines(
    lines: list[Line],
    img_h: int,
    img_w: int,
    edge_margins: Margins,
    y_min: int = 0,
    y_max: int | None = None,
    x_min: int = 0,
    x_max: int | None = None,
    aspect_ratio: float | None = None,
) -> FrameBounds:
    """Classify lines into edge groups based on position and orientation.

    Only considers lines near the edges of the image to avoid false positives
    from image content in the middle.

    When aspect_ratio is provided, uses the tight/loose approach to calculate
    classification thresholds so that the inner zone (where lines are ignored)
    has the correct aspect ratio.

    Args:
        lines: Detected Line objects
        img_h: Image height
        img_w: Image width
        edge_margins: Margins defining edge detection regions
        y_min: Minimum valid y coordinate (after sprocket cropping)
        y_max: Maximum valid y coordinate (after sprocket cropping)
        x_min: Minimum valid x coordinate (after sprocket cropping)
        x_max: Maximum valid x coordinate (after sprocket cropping)
        aspect_ratio: Target aspect ratio for ratio-aware threshold calculation

    Returns:
        FrameBounds with lines grouped by edge
    """
    if y_max is None:
        y_max = img_h
    if x_max is None:
        x_max = img_w

    frame_bounds = FrameBounds()

    # Calculate classification thresholds
    if aspect_ratio is not None:
        # Use ratio-aware margins: inner zone has correct aspect ratio
        edge_margin_percent = max(
            edge_margins.left, edge_margins.right,
            edge_margins.top, edge_margins.bottom
        )
        x_left_thresh, x_right_thresh, y_top_thresh, y_bottom_thresh = _calculate_ratio_aware_margins(
            img_w, img_h, aspect_ratio, edge_margin_percent,
            y_min, y_max, x_min, x_max
        )
    else:
        # Apply edge margins relative to the valid region (after sprocket cropping)
        valid_height = y_max - y_min
        valid_width = x_max - x_min
        y_top_thresh = y_min + int(valid_height * edge_margins.top)
        y_bottom_thresh = y_min + int(valid_height * (1 - edge_margins.bottom))
        x_left_thresh = x_min + int(valid_width * edge_margins.left)
        x_right_thresh = x_min + int(valid_width * (1 - edge_margins.right))

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
    x_min: int,
    x_max: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Adjust bounds to enforce the correct aspect ratio.

    Shrinks the larger dimension to match the expected aspect ratio,
    keeping the crop centered.

    Args:
        left, right, top, bottom: Current bounds
        aspect_ratio: Expected aspect ratio (width/height)
        y_min: Minimum valid y coordinate (sprocket boundary for horizontal)
        y_max: Maximum valid y coordinate (sprocket boundary for horizontal)
        x_min: Minimum valid x coordinate (sprocket boundary for vertical)
        x_max: Maximum valid x coordinate (sprocket boundary for vertical)
        img_w: Image width for clamping
        img_h: Image height for clamping

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

    # Re-clamp after adjustment (use appropriate boundaries based on orientation)
    # For horizontal orientation: clamp y to y_min/y_max, x to full width
    # For vertical orientation: clamp x to x_min/x_max, y to full height
    # We use both constraints for safety
    left = max(0, max(x_min, left)) if x_min > 0 else max(0, left)
    right = min(img_w, min(x_max, right)) if x_max < img_w else min(img_w, right)
    top = max(0, max(y_min, top)) if y_min > 0 else max(0, top)
    bottom = min(img_h, min(y_max, bottom)) if y_max < img_h else min(img_h, bottom)

    return left, right, top, bottom


def detect_frame_bounds(
    img: np.ndarray,
    aspect_ratio: float,
    visualizer: DebugVisualizer | None = None,
    crop_in_percent: float = 0.0,
    sprocket_margin_percent: float = 0.0,
    edge_margins: Margins | None = None,
    ignore_margins: Margins | None = None,
    film_base_inset_percent: float = 1.0,
    film_type: FilmType = FilmType.AUTO,
    edge_filter: EdgeFilter = EdgeFilter.CANNY,
    separation_method: SeparationMethod = SeparationMethod.COLOR_DISTANCE,
    filter_config: FilterConfig | None = None,
    preview_mode: str | None = None,
) -> tuple[FrameBounds, list[int]]:
    """Detect frame boundaries in an image.

    Args:
        img: Input image as numpy array
        aspect_ratio: Expected aspect ratio of the frame (width/height)
        visualizer: Optional debug visualizer to save intermediate images
        crop_in_percent: Percentage to crop inward from edges (0-100)
        sprocket_margin_percent: Additional percentage to crop beyond sprocket holes (0-100)
        edge_margins: Margins defining edge detection regions
        ignore_margins: Margins to crop before analysis
        film_base_inset_percent: Diagonal inset percentage for film base sampling (no sprockets)
        film_type: Type of film (NEGATIVE, POSITIVE, or AUTO for auto-detection)
        edge_filter: Edge detection filter method to use
        separation_method: Method to use for separating film base from image
        filter_config: Optional FilterConfig for advanced parameter control.
            If provided, overrides edge_filter and separation_method.
        preview_mode: Optional preview mode for minimal processing. 'separation' exits
            after film base mask visualization, 'edges' exits after edge detection.

    Returns:
        Tuple of (FrameBounds object, [left, right, top, bottom] positions)

    Raises:
        ValueError: If no lines or frame edges are detected, or invalid bounds
    """
    # If filter_config provided, extract settings from it
    if filter_config is not None:
        edge_filter = EdgeFilter(filter_config.edge_filter.method)
        separation_method = SeparationMethod(filter_config.separation.method)
        edge_filter_params = filter_config.edge_filter.get_params()
        separation_params = filter_config.separation.get_params()
        # Extract tolerance from separation params
        tolerance = separation_params.pop("tolerance", 30)
    else:
        edge_filter_params = None
        separation_params = None
        tolerance = 30
    if edge_margins is None:
        edge_margins = Margins(0.3, 0.3, 0.3, 0.3)
    if ignore_margins is None:
        ignore_margins = Margins(0.0, 0.0, 0.0, 0.0)

    orig_h, orig_w = img.shape[:2]
    img_h, img_w = orig_h, orig_w

    # Step 1: Detect sprocket holes on full image first
    sprocket_mask, detected_film_type = detect_sprocket_holes(img, film_type, visualizer)
    has_sprockets = detect_sprocket_presence(sprocket_mask, visualizer)

    if has_sprockets:
        orientation = detect_sprocket_orientation(sprocket_mask, visualizer)

        # Detect and filter out film cut ends before sprocket cropping
        cut_end = detect_film_cut_end(sprocket_mask, orientation, visualizer)
        sprocket_mask_filtered = filter_cut_end_from_sprocket_mask(
            sprocket_mask, orientation, cut_end
        )

        (
            _,
            y_offset_top,
            y_offset_bottom,
            x_offset_left,
            x_offset_right,
            sprocket_curve1,
            sprocket_curve2,
        ) = crop_sprocket_region(
            img, sprocket_mask_filtered, orientation, sprocket_margin_percent, visualizer
        )

        # Define valid frame region based on orientation
        y_min = y_offset_top
        y_max = img_h - y_offset_bottom
        x_min = x_offset_left
        x_max = img_w - x_offset_right

        if orientation == Orientation.HORIZONTAL:
            if y_min >= y_max:
                raise InvalidFrameRegionError(f"y_min={y_min} >= y_max={y_max}")
        else:
            if x_min >= x_max:
                raise InvalidFrameRegionError(f"x_min={x_min} >= x_max={x_max}")
    else:
        # No sprocket holes detected (e.g., medium format film)
        # Use full image as valid region
        orientation = Orientation.HORIZONTAL  # Default orientation
        y_min = 0
        y_max = img_h
        x_min = 0
        x_max = img_w
        sprocket_curve1 = None
        sprocket_curve2 = None
        cut_end = None

    # Apply ignore margins after sprocket detection
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

        # Adjust sprocket boundaries to account for ignore margins
        y_min = max(0, y_min - ignore_top)
        y_max = min(img_h, y_max - ignore_top)
        x_min = max(0, x_min - ignore_left)
        x_max = min(img_w, x_max - ignore_left)

        # Adjust sprocket curves to account for ignore margins
        if orientation == Orientation.HORIZONTAL:
            if sprocket_curve1 is not None:
                sprocket_curve1 = sprocket_curve1[ignore_left:orig_w - ignore_right] - ignore_top
            if sprocket_curve2 is not None:
                sprocket_curve2 = sprocket_curve2[ignore_left:orig_w - ignore_right] - ignore_top
        else:
            if sprocket_curve1 is not None:
                sprocket_curve1 = sprocket_curve1[ignore_top:orig_h - ignore_bottom] - ignore_left
            if sprocket_curve2 is not None:
                sprocket_curve2 = sprocket_curve2[ignore_top:orig_h - ignore_bottom] - ignore_left

        # Also crop the sprocket mask for film base detection
        sprocket_mask = sprocket_mask[
            ignore_top : orig_h - ignore_bottom,
            ignore_left : orig_w - ignore_right,
        ]

    # Step 2: Normalize levels for better color detection
    img = normalize_levels(img, visualizer)

    # Step 3: Detect film base color from normalized image
    film_base_color = detect_film_base_color(
        img,
        sprocket_mask,
        orientation,
        sprocket_curve1,
        sprocket_curve2,
        aspect_ratio,
        film_base_inset_percent,
        visualizer,
    )

    # Step 4: Create mask of film base regions
    film_base_mask = create_film_base_mask(
        img,
        film_base_color,
        tolerance=tolerance,
        visualizer=visualizer,
        separation_method=separation_method,
        separation_params=separation_params,
    )

    if visualizer:
        # Show film base mask with sprocket regions indicated
        film_base_mask_display = mask_sprocket_region(
            film_base_mask, orientation, sprocket_curve1, sprocket_curve2
        )
        visualizer.save_film_base_mask_cropped(
            img, film_base_mask_display, y_min, y_max, x_min, x_max, orientation,
            sprocket_curve1, sprocket_curve2
        )

    # Early exit for separation preview mode
    if preview_mode == "separation":
        return (
            FrameBounds(
                EdgeGroup([], None), EdgeGroup([], None),
                EdgeGroup([], None), EdgeGroup([], None)
            ),
            [0, img_w, 0, img_h],
        )

    # Step 5: Detect lines from film base mask boundaries (in each margin region)
    # Pass original mask - edges will be masked using curves inside detect_lines
    # Use ratio-aware margins so the inner zone has the correct aspect ratio
    lines, edges = detect_lines(
        img, film_base_mask, edge_margins, orientation,
        sprocket_curve1, sprocket_curve2, aspect_ratio,
        y_min, y_max, x_min, x_max, visualizer, edge_filter,
        cut_end, sprocket_margin_percent, edge_filter_params,
    )

    if visualizer:
        visualizer.save_lines(img, lines)

    if not lines:
        raise NoLinesDetectedError()

    # Calculate the actual margins used for visualization
    edge_margin_percent = max(
        edge_margins.left, edge_margins.right,
        edge_margins.top, edge_margins.bottom
    )
    left_margin, right_margin, top_margin, bottom_margin = _calculate_ratio_aware_margins(
        img_w, img_h, aspect_ratio, edge_margin_percent,
        y_min, y_max, x_min, x_max
    )

    if visualizer:
        visualizer.save_edge_margins(img, left_margin, right_margin, top_margin, bottom_margin)

    frame_bounds = classify_lines(
        lines, img_h, img_w, edge_margins, y_min, y_max, x_min, x_max, aspect_ratio
    )

    if visualizer:
        # Pass img for overlay in preview mode
        visualizer.save_edges(edges, frame_bounds, img=img if preview_mode == "edges" else None)
        visualizer.save_classified_lines(img, frame_bounds)

    # Early exit for edges preview mode (after classification)
    if preview_mode == "edges":
        return (frame_bounds, [0, img_w, 0, img_h])

    if not frame_bounds.top.lines and not frame_bounds.bottom.lines:
        if not frame_bounds.left.lines and not frame_bounds.right.lines:
            raise NoFrameEdgesError()

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
    # Use sprocket boundaries to infer missing edge positions
    has_sprockets = sprocket_curve1 is not None or sprocket_curve2 is not None

    if left is None or right is None or top is None or bottom is None:
        # Estimate frame size from what we have
        if has_top and has_bottom:
            height = bottom - top
            expected_width = int(height * aspect_ratio)
        elif has_left and has_right:
            expected_width = right - left
        else:
            # Use the valid region based on sprocket boundaries
            if has_sprockets:
                if orientation == Orientation.HORIZONTAL:
                    # Frame height is determined by sprocket boundaries
                    height = y_max - y_min
                else:
                    # Frame width is determined by sprocket boundaries
                    height = x_max - x_min
            else:
                # No sprockets - use 90% of image
                if orientation == Orientation.HORIZONTAL:
                    height = int(img_h * 0.9)
                else:
                    height = int(img_h * 0.9)
            expected_width = int(height * aspect_ratio)

        if left is None and right is None:
            if orientation == Orientation.VERTICAL and has_sprockets:
                # Vertical film with sprockets: center horizontally between sprocket boundaries
                center_x = (x_min + x_max) // 2
                left = center_x - expected_width // 2
                right = center_x + expected_width // 2
            else:
                # Center in the image
                left = (img_w - expected_width) // 2
                right = (img_w + expected_width) // 2
        elif left is None:
            left = right - expected_width
        elif right is None:
            right = left + expected_width

        expected_height = int((right - left) / aspect_ratio)
        if top is None and bottom is None:
            if orientation == Orientation.HORIZONTAL and has_sprockets:
                # Horizontal film with sprockets: center vertically between sprocket boundaries
                center_y = (y_min + y_max) // 2
                top = center_y - expected_height // 2
                bottom = center_y + expected_height // 2
            else:
                # Center in the image
                center_y = img_h // 2
                top = center_y - expected_height // 2
                bottom = center_y + expected_height // 2
        elif top is None:
            top = bottom - expected_height
        elif bottom is None:
            bottom = top + expected_height

    # Clamp coordinates to image bounds (excluding sprocket regions)
    if orientation == Orientation.HORIZONTAL:
        left = max(0, left)
        right = min(img_w, right)
        top = max(y_min, top)
        bottom = min(y_max, bottom)
    else:
        left = max(x_min, left)
        right = min(x_max, right)
        top = max(0, top)
        bottom = min(img_h, bottom)

    if right <= left or bottom <= top:
        raise InvalidFrameCoordinatesError()

    # Apply crop-in to exclude film base from final crop
    left, right, top, bottom = apply_crop_in(
        img, film_base_mask, left, right, top, bottom, crop_in_percent, visualizer
    )

    # Enforce correct aspect ratio
    left, right, top, bottom = enforce_aspect_ratio(
        left, right, top, bottom, aspect_ratio, y_min, y_max, x_min, x_max, img_w, img_h
    )

    if visualizer:
        visualizer.save_bounds(img, [left, right, top, bottom])

    # Adjust bounds back to original image coordinates
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
