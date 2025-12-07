"""Edge detection filter methods for frame boundary detection."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class EdgeFilter(Enum):
    """Available edge detection filter methods."""

    CANNY = "canny"
    SOBEL = "sobel"
    SCHARR = "scharr"
    DOG = "dog"
    LAPLACIAN = "laplacian"
    LOG = "log"


def apply_canny(mask: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Apply Canny edge detection.

    Args:
        mask: Input grayscale image or binary mask
        low: Lower threshold for hysteresis
        high: Upper threshold for hysteresis

    Returns:
        Binary edge map (0 or 255)
    """
    return cv2.Canny(mask, low, high)


def apply_sobel(gray: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """Apply Sobel gradient magnitude + Gaussian blur.

    Args:
        gray: Input grayscale image
        blur_size: Gaussian blur kernel size (will be made odd)

    Returns:
        Binary edge map (0 or 255)
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to 0-255
    if gradient.max() > 0:
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    else:
        gradient = gradient.astype(np.uint8)

    # Apply blur
    if blur_size > 0:
        ksize = blur_size | 1  # Ensure odd
        gradient = cv2.GaussianBlur(gradient, (ksize, ksize), 0)

    # Threshold to binary
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_scharr(gray: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """Apply Scharr gradient magnitude + Gaussian blur.

    Scharr is more accurate than Sobel for detecting fine edges.

    Args:
        gray: Input grayscale image
        blur_size: Gaussian blur kernel size (will be made odd)

    Returns:
        Binary edge map (0 or 255)
    """
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    gradient = np.sqrt(scharr_x**2 + scharr_y**2)

    # Normalize to 0-255
    if gradient.max() > 0:
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)
    else:
        gradient = gradient.astype(np.uint8)

    # Apply blur
    if blur_size > 0:
        ksize = blur_size | 1  # Ensure odd
        gradient = cv2.GaussianBlur(gradient, (ksize, ksize), 0)

    # Threshold to binary
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_dog(gray: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0) -> np.ndarray:
    """Apply Difference of Gaussians filter.

    DoG enhances edges with built-in smoothing by subtracting two
    differently blurred versions of the image.

    Args:
        gray: Input grayscale image
        sigma1: Sigma for first (smaller) Gaussian
        sigma2: Sigma for second (larger) Gaussian

    Returns:
        Binary edge map (0 or 255)
    """
    ksize1 = int(sigma1 * 6) | 1
    ksize2 = int(sigma2 * 6) | 1

    blur1 = cv2.GaussianBlur(gray, (ksize1, ksize1), sigma1)
    blur2 = cv2.GaussianBlur(gray, (ksize2, ksize2), sigma2)

    dog = blur1.astype(np.float64) - blur2.astype(np.float64)

    # Normalize to 0-255
    dog = dog - dog.min()
    if dog.max() > 0:
        dog = (dog / dog.max() * 255).astype(np.uint8)
    else:
        dog = dog.astype(np.uint8)

    # Threshold to binary
    _, binary = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_laplacian(gray: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """Apply Laplacian filter + Gaussian blur.

    Args:
        gray: Input grayscale image
        blur_size: Gaussian blur kernel size (will be made odd)

    Returns:
        Binary edge map (0 or 255)
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian)

    # Normalize to 0-255
    if laplacian.max() > 0:
        laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)
    else:
        laplacian = laplacian.astype(np.uint8)

    # Apply blur
    if blur_size > 0:
        ksize = blur_size | 1  # Ensure odd
        laplacian = cv2.GaussianBlur(laplacian, (ksize, ksize), 0)

    # Threshold to binary
    _, binary = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_log(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Laplacian of Gaussian (LoG) filter.

    Applies Gaussian blur first, then Laplacian. This is more robust
    to noise than applying Laplacian directly.

    Args:
        gray: Input grayscale image
        sigma: Sigma for Gaussian blur

    Returns:
        Binary edge map (0 or 255)
    """
    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = np.abs(log)

    # Normalize to 0-255
    if log.max() > 0:
        log = (log / log.max() * 255).astype(np.uint8)
    else:
        log = log.astype(np.uint8)

    # Threshold to binary
    _, binary = cv2.threshold(log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# Mapping from EdgeFilter enum to filter function
_FILTER_FUNCTIONS = {
    EdgeFilter.CANNY: apply_canny,
    EdgeFilter.SOBEL: apply_sobel,
    EdgeFilter.SCHARR: apply_scharr,
    EdgeFilter.DOG: apply_dog,
    EdgeFilter.LAPLACIAN: apply_laplacian,
    EdgeFilter.LOG: apply_log,
}


def apply_filter(mask: np.ndarray, edge_filter: EdgeFilter) -> np.ndarray:
    """Apply the specified edge detection filter.

    Args:
        mask: Input grayscale image or binary mask
        edge_filter: Which filter method to use

    Returns:
        Binary edge map (0 or 255)
    """
    filter_fn = _FILTER_FUNCTIONS[edge_filter]
    return filter_fn(mask)


def apply_all_filters(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Apply all edge detection filters and return results.

    Args:
        mask: Input grayscale image or binary mask

    Returns:
        Dictionary mapping filter name to edge map result
    """
    results = {}
    for edge_filter in EdgeFilter:
        results[edge_filter.value] = apply_filter(mask, edge_filter)
    return results
