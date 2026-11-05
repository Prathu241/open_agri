import unittest

from PIL import Image

from image_quality import assess_image_quality


class ImageQualityTests(unittest.TestCase):
    def test_rejects_tiny_dark_image(self) -> None:
        result = assess_image_quality(Image.new("RGB", (64, 64), "black"))
        self.assertFalse(result["accepted"])
        self.assertGreaterEqual(len(result["reasons"]), 3)

    def test_accepts_detailed_well_lit_image(self) -> None:
        image = Image.new("RGB", (320, 320))
        pixels = image.load()
        for y in range(320):
            for x in range(320):
                value = 20 if (x // 8 + y // 8) % 2 else 235
                pixels[x, y] = (value, value, value)
        result = assess_image_quality(image)
        self.assertTrue(result["accepted"], result["reasons"])


if __name__ == "__main__":
    unittest.main()
