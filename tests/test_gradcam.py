import os
import unittest

import numpy as np
from PIL import Image

from gradcam.attention_map import get_attention_map


class TestGradCAM(unittest.TestCase):
    def setUp(self):
        self.caption_text = "a cat"
        self.image_path = "test.jpg"
        test_data = np.random.rand(1080, 1920, 3) * 255
        test_img = Image.fromarray(test_data.astype("uint8")).convert("RGB")
        test_img.save(self.image_path)

    def tearDown(self):
        os.remove(self.image_path)

    def test_pretrained(self):
        for model_name, pretrain_tag in [
            ("RN50", "openai"),
            ("RN101", "openai"),
            ("convnext_base", "laion400m_s13b_b51k"),
            ("convnext_base_w", "laion2b_s13b_b82k"),
            ("ViT-B-16", "openai"),
            ("ViT-B-32", "openai"),
            ("ViT-L-14", "openai"),
        ]:
            attention_map = get_attention_map(
                self.image_path, self.caption_text, model_name, pretrain_tag
            )
            assert isinstance(attention_map, np.ndarray)

    def test_hf_hub(self):
        for model_name in ["hf-hub:timm/ViT-B-16-SigLIP"]:
            attention_map = get_attention_map(
                self.image_path, self.caption_text, model_name
            )
            assert isinstance(attention_map, np.ndarray)


if __name__ == "__main__":
    unittest.main()
