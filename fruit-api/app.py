"""Production-shaped research API for Jeevandhara public-fruit-v1."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from image_quality import assess_image_quality


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FRUIT_MODEL_PATH", ROOT / "model/model.pt"))
METADATA_PATH = Path(os.environ.get("FRUIT_METADATA_PATH", ROOT / "model/metadata.json"))
CALIBRATION_PATH = Path(
    os.environ.get("FRUIT_CALIBRATION_PATH", ROOT / "model/calibration.json")
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_CROPS = {
    "apple",
    "banana",
    "blueberry",
    "cherry",
    "grape",
    "peach",
    "strawberry",
}

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
model: torch.jit.ScriptModule | None = None

app = FastAPI(
    title="Jeevandhara Public Fruit Research API",
    version="1.1.0",
    description="Calibrated research-only crop-condition predictions with quality and crop gates.",
)


def get_model() -> torch.jit.ScriptModule:
    global model
    if model is None:
        model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    scale = 256.0 / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
    )
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(metadata["normalization"]["std"], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def infer(image: Image.Image, top_k: int = 3) -> dict[str, Any]:
    with torch.inference_mode():
        logits = get_model()(preprocess(image)).float()
        probabilities = torch.softmax(logits / float(calibration["temperature"]), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, len(metadata["labels"])))
    candidates = []
    for confidence, index in zip(values.tolist(), indices.tolist()):
        label = metadata["labels"][index]
        crop, condition = label.split("__", 1)
        candidates.append(
            {
                "label": label,
                "crop": crop,
                "condition": condition,
                "confidence": confidence,
            }
        )
    threshold = float(calibration["uncertainty"]["threshold"])
    top = candidates[0]
    uncertain = top["confidence"] < threshold
    return {
        "model_version": calibration["model_version"],
        "research_only": True,
        "status": "uncertain" if uncertain else "candidate_found",
        "uncertainty_reason": (
            f"Top confidence {top['confidence']:.3f} is below threshold {threshold:.3f}."
            if uncertain
            else None
        ),
        "prediction": None if uncertain else top,
        "candidates": candidates,
        "confidence_threshold": threshold,
    }


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "jeevandhara-fruit-api",
        "version": app.version,
        "docs": "/docs",
        "research_only": True,
    }


@app.get("/health")
def health() -> dict[str, object]:
    loaded_model = get_model()
    return {
        "status": "ready",
        "model_version": calibration["model_version"],
        "device": "cpu",
        "classes": len(metadata["labels"]),
        "model_loaded": loaded_model is not None,
        "research_only": True,
    }


@app.post("/v1/predict")
async def predict(
    file: UploadFile = File(...),
    expected_crop: str | None = Form(default=None),
) -> dict[str, object]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload JPEG, PNG, or WebP.")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    normalized_crop = expected_crop.strip().lower() if expected_crop else None
    if normalized_crop and normalized_crop not in SUPPORTED_CROPS:
        supported = ", ".join(sorted(SUPPORTED_CROPS))
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported crop '{expected_crop}'. Select one of: {supported}.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            quality = assess_image_quality(image)
            if quality["accepted"]:
                result = infer(image)
            else:
                result = {
                    "model_version": calibration["model_version"],
                    "research_only": True,
                    "status": "poor_quality",
                    "uncertainty_reason": " ".join(quality["reasons"]),
                    "prediction": None,
                    "candidates": [],
                    "confidence_threshold": calibration["uncertainty"]["threshold"],
                }
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The upload is not a readable image.") from exc

    result["expected_crop"] = normalized_crop
    result["quality"] = quality
    prediction = result.get("prediction")
    if (
        normalized_crop
        and isinstance(prediction, dict)
        and prediction.get("crop") != normalized_crop
    ):
        predicted_crop = prediction.get("crop", "another crop")
        result["status"] = "crop_mismatch"
        result["prediction"] = None
        result["uncertainty_reason"] = (
            f"The model's top candidate was {predicted_crop}, not the selected {normalized_crop}."
        )
    result["advisory"] = None
    result["disclaimer"] = (
        "Research result only. Do not apply treatment or pesticide decisions without local expert review."
    )
    return result
