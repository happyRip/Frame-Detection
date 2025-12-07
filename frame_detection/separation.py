"""Film base separation methods for improved frame detection."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class SeparationMethod(Enum):
    """Available film base separation methods."""

    COLOR_DISTANCE = "color_distance"  # Current method - Euclidean distance in BGR
    CLAHE = "clahe"  # Local contrast enhancement
    LAB_DISTANCE = "lab_distance"  # Distance in LAB color space
    HSV_DISTANCE = "hsv_distance"  # Distance in HSV color space
    ADAPTIVE = "adaptive"  # Adaptive thresholding on color distance
    GRADIENT = "gradient"  # Gradient-enhanced separation


def apply_color_distance(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
) -> np.ndarray:
    """Original method - Euclidean distance in BGR color space.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching

    Returns:
        Binary mask where film base regions are 255
    """
    diff = img.astype(np.float32) - film_base_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff**2, axis=2))
    mask = (distance <= tolerance).astype(np.uint8) * 255
    return mask


def apply_clahe(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
    clip_limit: float = 1.0,
    tile_size: int = 32,
) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement before color matching.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
    local contrast, which can help separate film base from image content
    when they have similar global brightness.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_size: CLAHE tile grid size

    Returns:
        Binary mask where film base regions are 255
    """
    # Convert to LAB for CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)

    # Reconstruct image
    lab_enhanced = lab.copy()
    lab_enhanced[:, :, 0] = l_enhanced
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Also enhance the film base color reference (approximate)
    # by applying same contrast stretch
    l_base = cv2.cvtColor(
        film_base_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
    )[0, 0, 0]
    # Simple linear mapping based on original vs enhanced range
    l_min, l_max = l_channel.min(), l_channel.max()
    l_enh_min, l_enh_max = l_enhanced.min(), l_enhanced.max()
    if l_max > l_min:
        l_base_enhanced = int(
            (l_base - l_min) / (l_max - l_min) * (l_enh_max - l_enh_min) + l_enh_min
        )
    else:
        l_base_enhanced = l_base

    # Reconstruct enhanced film base color
    lab_base = cv2.cvtColor(film_base_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)
    lab_base[0, 0, 0] = l_base_enhanced
    film_base_enhanced = cv2.cvtColor(lab_base, cv2.COLOR_LAB2BGR)[0, 0]

    # Apply color distance on enhanced image
    diff = img_enhanced.astype(np.float32) - film_base_enhanced.astype(np.float32)
    distance = np.sqrt(np.sum(diff**2, axis=2))
    mask = (distance <= tolerance).astype(np.uint8) * 255
    return mask


def apply_lab_distance(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
) -> np.ndarray:
    """Distance in LAB color space for perceptually uniform separation.

    LAB color space is designed to be perceptually uniform, meaning
    equal distances correspond to equal perceived color differences.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching

    Returns:
        Binary mask where film base regions are 255
    """
    # Convert image and film base to LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    film_base_lab = cv2.cvtColor(
        film_base_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]

    # Compute distance in LAB space
    diff = img_lab - film_base_lab
    distance = np.sqrt(np.sum(diff**2, axis=2))

    # LAB distances are typically larger, adjust tolerance
    lab_tolerance = tolerance * 1.5
    mask = (distance <= lab_tolerance).astype(np.uint8) * 255
    return mask


def apply_hsv_distance(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
) -> np.ndarray:
    """Distance in HSV color space, weighted by saturation and value.

    HSV separates color (hue) from intensity (value), which can help
    when film base has a distinct color cast (like orange negative base).

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching

    Returns:
        Binary mask where film base regions are 255
    """
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    film_base_hsv = cv2.cvtColor(
        film_base_color.reshape(1, 1, 3), cv2.COLOR_BGR2HSV
    ).astype(np.float32)[0, 0]

    # Hue is circular (0-180 in OpenCV), need special handling
    h_diff = np.abs(img_hsv[:, :, 0] - film_base_hsv[0])
    h_diff = np.minimum(h_diff, 180 - h_diff)  # Handle wraparound

    # S and V differences
    s_diff = np.abs(img_hsv[:, :, 1] - film_base_hsv[1])
    v_diff = np.abs(img_hsv[:, :, 2] - film_base_hsv[2])

    # Weighted distance (hue matters less for low saturation)
    saturation_weight = img_hsv[:, :, 1] / 255.0
    distance = np.sqrt(
        (h_diff * saturation_weight * 2) ** 2 + s_diff**2 + v_diff**2
    )

    # HSV distances have different scale
    hsv_tolerance = tolerance * 2
    mask = (distance <= hsv_tolerance).astype(np.uint8) * 255
    return mask


def apply_adaptive(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
    block_size: int = 51,
) -> np.ndarray:
    """Adaptive thresholding on color distance map.

    Instead of a fixed tolerance, uses local adaptive thresholding
    which can handle varying lighting across the image.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Base tolerance (used to scale adaptive threshold)
        block_size: Block size for adaptive threshold (must be odd)

    Returns:
        Binary mask where film base regions are 255
    """
    # Compute color distance
    diff = img.astype(np.float32) - film_base_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff**2, axis=2))

    # Normalize distance to 0-255 for thresholding
    dist_max = distance.max()
    if dist_max > 0:
        distance_norm = (distance / dist_max * 255).astype(np.uint8)
    else:
        distance_norm = np.zeros(distance.shape, dtype=np.uint8)

    # Adaptive threshold - areas close to film base will be bright
    # We want to threshold on "closeness" so invert
    closeness = 255 - distance_norm

    # Ensure block_size is odd
    block_size = block_size | 1

    # Adaptive threshold
    mask = cv2.adaptiveThreshold(
        closeness,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -5,  # C constant (negative = more lenient)
    )

    return mask


def apply_gradient(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
    gradient_weight: float = 0.5,
) -> np.ndarray:
    """Gradient-enhanced separation using edge information.

    Combines color distance with gradient magnitude to enhance
    boundaries between film base and image content.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching
        gradient_weight: Weight for gradient contribution (0-1)

    Returns:
        Binary mask where film base regions are 255
    """
    # Compute color distance
    diff = img.astype(np.float32) - film_base_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff**2, axis=2))

    # Compute gradient magnitude on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient to 0-1
    grad_max = gradient.max()
    if grad_max > 0:
        gradient_norm = gradient / grad_max
    else:
        gradient_norm = np.zeros_like(gradient)

    # Increase distance at high-gradient areas (edges)
    # This helps separate film base from image even when colors are similar
    enhanced_distance = distance * (1 + gradient_norm * gradient_weight * 2)

    # Apply threshold
    mask = (enhanced_distance <= tolerance).astype(np.uint8) * 255
    return mask


# Mapping from SeparationMethod enum to function
_SEPARATION_FUNCTIONS = {
    SeparationMethod.COLOR_DISTANCE: apply_color_distance,
    SeparationMethod.CLAHE: apply_clahe,
    SeparationMethod.LAB_DISTANCE: apply_lab_distance,
    SeparationMethod.HSV_DISTANCE: apply_hsv_distance,
    SeparationMethod.ADAPTIVE: apply_adaptive,
    SeparationMethod.GRADIENT: apply_gradient,
}


def apply_separation(
    img: np.ndarray,
    film_base_color: np.ndarray,
    method: SeparationMethod,
    tolerance: int = 30,
) -> np.ndarray:
    """Apply the specified separation method.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        method: Which separation method to use
        tolerance: Color distance tolerance for matching

    Returns:
        Binary mask where film base regions are 255
    """
    fn = _SEPARATION_FUNCTIONS[method]
    return fn(img, film_base_color, tolerance)


def apply_all_separations(
    img: np.ndarray,
    film_base_color: np.ndarray,
    tolerance: int = 30,
) -> dict[str, np.ndarray]:
    """Apply all separation methods and return results.

    Args:
        img: Input BGR image
        film_base_color: BGR color of the film base
        tolerance: Color distance tolerance for matching

    Returns:
        Dictionary mapping method name to mask result
    """
    results = {}
    for method in SeparationMethod:
        results[method.value] = apply_separation(img, film_base_color, method, tolerance)
    return results
