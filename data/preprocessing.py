"""
NeuroScript AI - Image Preprocessing Pipeline
Provides mathematically sound and consistent preprocessing for:
- Drawing canvas inputs (light canvas, dark strokes)
- Uploaded images (photographs, scans, screenshots)
- Dataset images (MNIST / EMNIST)
Follows standard LeCun MNIST centering:
1. Grayscale conversion
2. Polarity normalization (stroke=bright, background=dark)
3. Content bounding box extraction
4. Aspect-ratio preserving scaling to 20x20
5. Centering within 28x28 canvas using center of mass / geometric center
6. Normalization to [0.0, 1.0]
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
from config.settings import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, ARCH_ANN, ARCH_CNN


class ImagePreprocessor:
    """
    Standardized preprocessing engine for handwriting recognition models.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
        inner_box_size: int = 20,
        empty_threshold: float = 0.02
    ):
        self.target_width, self.target_height = target_size
        self.inner_box_size = inner_box_size
        self.empty_threshold = empty_threshold

    def is_image_empty(self, gray_array: np.ndarray, stroke_threshold: int = 30) -> bool:
        """
        Check whether an image contains no detectable stroke content.
        Assumes normalized stroke where foreground > background.
        """
        if gray_array is None or gray_array.size == 0:
            return True
        non_zero = np.count_nonzero(gray_array > stroke_threshold)
        total_pixels = gray_array.size
        ratio = non_zero / float(total_pixels)
        return bool(ratio < self.empty_threshold and np.max(gray_array) < stroke_threshold)

    def normalize_polarity(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Ensure strokes are white (high values) and background is black (low values),
        matching standard MNIST/EMNIST training distribution.
        """
        # Sample border pixels to detect background color
        h, w = img_gray.shape
        border_pixels = np.concatenate([
            img_gray[0, :],
            img_gray[h - 1, :],
            img_gray[:, 0],
            img_gray[:, w - 1]
        ])
        border_mean = np.mean(border_pixels)

        # If background is bright (white canvas / paper), invert the image
        if border_mean > 127:
            return 255 - img_gray
        return img_gray

    def extract_bounding_box(self, img_gray: np.ndarray, thresh_val: int = 40) -> Optional[np.ndarray]:
        """
        Extract the tight bounding box containing the stroke content.
        Returns None if image is empty.
        """
        # Threshold to separate stroke from background noise
        _, thresh = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find encompassing bounding box of all strokes
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        has_content = False

        for c in contours:
            if cv2.contourArea(c) > 2:  # Filter minor noise
                x, y, w, h = cv2.boundingRect(c)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
                has_content = True

        if not has_content or x_max <= x_min or y_max <= y_min:
            return None

        return img_gray[int(y_min):int(y_max), int(x_min):int(x_max)]

    def center_and_pad(self, cropped: np.ndarray) -> np.ndarray:
        """
        Resize cropped digit to fit within inner_box_size (e.g. 20x20) preserving aspect ratio,
        and place it centered in a (target_height, target_width) canvas.
        """
        h, w = cropped.shape
        if h == 0 or w == 0:
            return np.zeros((self.target_height, self.target_width), dtype=np.uint8)

        # Scale factor so max dimension equals inner_box_size
        scale = float(self.inner_box_size) / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # High-quality interpolation
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create blank 28x28 black canvas
        canvas = np.zeros((self.target_height, self.target_width), dtype=np.uint8)

        # Calculate coordinates to center by geometric center
        start_x = (self.target_width - new_w) // 2
        start_y = (self.target_height - new_h) // 2

        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        # Calculate center of mass shift (LeCun refinement)
        moments = cv2.moments(canvas)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            shift_x = int(round(self.target_width / 2.0 - cx))
            shift_y = int(round(self.target_height / 2.0 - cy))

            # Limit shift to avoid pushing stroke out of canvas
            shift_x = max(-3, min(3, shift_x))
            shift_y = max(-3, min(3, shift_y))

            m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, m, (self.target_width, self.target_height))

        return canvas

    def preprocess_image(
        self,
        image_input: Any,
        architecture: str = ARCH_CNN
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Preprocess input image (PIL Image, numpy array, or file-like object).
        Returns:
            processed_tensor: normalized tensor (1, 28, 28, 1) for CNN or (1, 784) for ANN.
            metadata: dictionary containing diagnostic info (is_empty, intermediate arrays).
        """
        metadata = {
            "is_empty": False,
            "raw_shape": None,
            "original_gray": None,
            "polarity_adjusted": None,
            "normalized_28x28": None
        }

        # Convert to numpy array
        if isinstance(image_input, Image.Image):
            img_arr = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            img_arr = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        metadata["raw_shape"] = img_arr.shape

        # Convert to Grayscale
        if len(img_arr.shape) == 3:
            if img_arr.shape[2] == 4:  # RGBA
                # Handle alpha channel
                alpha = img_arr[:, :, 3]
                rgb = img_arr[:, :, :3]
                # If transparent background with white/black strokes
                if np.all(alpha == 0):
                    metadata["is_empty"] = True
                    return None, metadata
                # Blend with white background before converting
                bg = np.ones_like(rgb, dtype=np.uint8) * 255
                alpha_factor = alpha[:, :, np.newaxis] / 255.0
                blended = (rgb * alpha_factor + bg * (1.0 - alpha_factor)).astype(np.uint8)
                gray = cv2.cvtColor(blended, cv2.COLOR_RGB2GRAY)
            elif img_arr.shape[2] == 3:  # RGB
                gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_arr[:, :, 0]
        elif len(img_arr.shape) == 2:
            gray = img_arr
        else:
            raise ValueError(f"Invalid image array shape: {img_arr.shape}")

        metadata["original_gray"] = gray

        # Normalize polarity (strokes bright, background dark)
        polarity_adjusted = self.normalize_polarity(gray)
        metadata["polarity_adjusted"] = polarity_adjusted

        # Check for empty content
        if self.is_image_empty(polarity_adjusted):
            metadata["is_empty"] = True
            return None, metadata

        # Crop to stroke bounding box
        cropped = self.extract_bounding_box(polarity_adjusted)
        if cropped is None:
            metadata["is_empty"] = True
            return None, metadata

        # Center in 28x28
        centered = self.center_and_pad(cropped)
        metadata["normalized_28x28"] = centered

        # Normalize to [0.0, 1.0] float32
        norm_float = centered.astype(np.float32) / 255.0

        # Reshape according to architecture
        if architecture == ARCH_CNN:
            # (1, 28, 28, 1)
            tensor = np.expand_dims(norm_float, axis=(0, -1))
        elif architecture == ARCH_ANN:
            # (1, 784)
            tensor = norm_float.reshape(1, -1)
        else:
            tensor = np.expand_dims(norm_float, axis=(0, -1))

        return tensor, metadata


def get_default_preprocessor() -> ImagePreprocessor:
    """Return preprocessor with default LeCun parameters."""
    return ImagePreprocessor()
