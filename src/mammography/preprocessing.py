"""Lossless mammogram preprocessing functions."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import wiener


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Robustly map an image to the 8-bit range."""

    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    array = array.astype(np.float32)
    low, high = np.percentile(array, (0.5, 99.5))
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = np.clip((array - low) / (high - low), 0, 1)
    return np.round(normalized * 255).astype(np.uint8)


def add_gaussian_noise(
    image: np.ndarray, *, variance: float = 0.002, seed: int | None = None
) -> np.ndarray:
    """Add reproducible Gaussian noise for controlled robustness experiments."""

    if variance < 0:
        raise ValueError("variance must be non-negative")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(variance) * 255.0, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def denoise(image: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "none":
        return image
    if method == "median":
        return cv2.medianBlur(image, 5)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, 50, 50)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    if method == "wiener":
        filtered = wiener(image.astype(np.float32), (5, 5))
        return np.clip(filtered, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown denoising method: {method}")


def enhance_contrast(image: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "none":
        return image
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        return clahe.apply(image)
    if method == "histogram":
        return cv2.equalizeHist(image)
    raise ValueError(f"Unknown enhancement method: {method}")


def preprocess_image(
    image: np.ndarray,
    *,
    denoise_method: str = "none",
    enhancement_method: str = "clahe",
    noise_variance: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    output = to_uint8(image)
    if noise_variance is not None:
        output = add_gaussian_noise(output, variance=noise_variance, seed=seed)
    output = denoise(output, denoise_method)
    return enhance_contrast(output, enhancement_method)


def canny_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150):
    return cv2.Canny(to_uint8(image), low_threshold, high_threshold)
