import os
import tempfile
import requests
import unittest
import numpy as np

from PIL import Image


is_pp_api_available = False
try:
    from transformers import AutoImageProcessor
    from onnxruntime_extensions import pp_api
    is_pp_api_available = True
except ImportError:
    pass


def regen_image(arr):
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])

    # Reverse normalization
    array = arr * std + mean

    # Clip the values to [0, 1] range
    array = np.clip(array, 0, 1)

    # Convert to [0, 255] range and uint8 type
    array = (array * 255).astype(np.uint8)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(array)
    return image


@unittest.skipIf(not is_pp_api_available, "pp_api is not available")
class TestPPAPI(unittest.TestCase):
    def test_llama3_2_image_processing(self):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        processor = AutoImageProcessor.from_pretrained(model_id)

        url = ("https://huggingface.co/datasets/huggingface/"
               "documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg")
        image = Image.open(requests.get(url, stream=True).raw)
        image2 = Image.open("test/data/processor/passport.png")
        image3 = Image.open("test/data/processor/exceltable.png")
        inputs = processor.preprocess([image, image2, image3], return_tensors="np")
        print({k: v.shape if k == "pixel_values" else v for k, v in inputs.items()})

        if os.path.exists("/temp"):
            temp_dir = "/temp"
        else:
            temp_dir = tempfile.mkdtemp()
            print(f"Created temp dir: {temp_dir}")

        expected_images = inputs["pixel_values"][0][1]
        for i in range(len(expected_images)):
            expected = expected_images[i]
            e_image = regen_image(np.transpose(expected, (1, 2, 0)))
            e_image.save(f"{temp_dir}/e_{i}.png")


if __name__ == '__main__':
    unittest.main()
