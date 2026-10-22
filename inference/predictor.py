"""
NeuroScript AI - Inference Predictor Engine
Coordinates preprocessing, neural network inference, confidence scoring,
top-k ranking, and diagnostic previews for drawing canvas and uploaded images.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import tensorflow as tf
from data.preprocessing import ImagePreprocessor, get_default_preprocessor
from data.labels import LabelMapper
from config.settings import ARCH_ANN, ARCH_CNN


class CharacterPredictor:
    """
    Executes inference using a trained ANN or CNN model.
    """

    def __init__(self, preprocessor: Optional[ImagePreprocessor] = None):
        self.preprocessor = preprocessor or get_default_preprocessor()

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray],
        model: tf.keras.Model,
        task: str,
        architecture: str,
        top_k_count: int = 5
    ) -> Dict[str, Any]:
        """
        Run inference pipeline on an input image.
        Returns:
            Dictionary containing prediction result, confidence, top-k classes,
            preprocessed 28x28 image, and execution diagnostics.
        """
        arch = architecture.upper()
        mapper = LabelMapper(task)
        classes = mapper.get_classes()

        # Step 1: Preprocess Image
        tensor, meta = self.preprocessor.preprocess_image(
            image_input=image_input,
            architecture=arch
        )

        if meta.get("is_empty") or tensor is None:
            return {
                "success": False,
                "error": "EMPTY_INPUT",
                "message": "Input is blank. Please draw a stroke or upload an image containing a handwritten character.",
                "preprocessed_image": None
            }

        # Step 2: Model Inference
        start_time = time.perf_counter()
        probabilities = model.predict(tensor, verbose=0)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Step 3: Parse Predictions
        pred_idx = int(np.argmax(probabilities))
        pred_confidence = float(probabilities[pred_idx])
        pred_label = mapper.index_to_label(pred_idx)

        # Step 4: Top-K Ranking
        top_indices = np.argsort(probabilities)[::-1][:min(top_k_count, len(classes))]
        top_k = []
        for idx in top_indices:
            top_k.append({
                "class": mapper.index_to_label(int(idx)),
                "index": int(idx),
                "probability": float(probabilities[idx]),
                "percentage": f"{probabilities[idx] * 100.0:.2f}%"
            })

        return {
            "success": True,
            "predicted_class": pred_label,
            "predicted_index": pred_idx,
            "confidence": pred_confidence,
            "confidence_percentage": f"{pred_confidence * 100.0:.2f}%",
            "top_k": top_k,
            "all_probabilities": probabilities.tolist(),
            "preprocessed_image": meta.get("normalized_28x28"),
            "inference_time_ms": inference_time_ms,
            "task": task,
            "architecture": arch
        }


_predictor = None


def get_predictor() -> CharacterPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CharacterPredictor()
    return _predictor
