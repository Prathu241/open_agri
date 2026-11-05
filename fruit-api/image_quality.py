"""Conservative image-quality checks before fruit-model inference."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageFilter, ImageStat


def assess_image_quality(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    grayscale = rgb.convert("L")
    grayscale.thumbnail((512, 512))
    stats = ImageStat.Stat(grayscale)
    brightness = float(stats.mean[0])
    contrast = float(stats.stddev[0])
    edge_variance = float(ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES)).var[0])

    reasons: list[str] = []
    if min(width, height) < 224:
        reasons.append(f"Image is too small ({width}x{height}); use at least 224px on each side.")
    if brightness < 20:
        reasons.append("Image is too dark; retake it in natural light.")
    elif brightness > 245:
        reasons.append("Image is overexposed; avoid glare and direct flash.")
    if contrast < 8:
        reasons.append("Image has too little visible detail; move closer and refocus.")
    if edge_variance < 4:
        reasons.append("Image appears blurred or featureless; hold the phone steady and refocus.")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "metrics": {
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "edge_variance": round(edge_variance, 2),
        },
        "policy_version": "quality-gate-v1",
    }
