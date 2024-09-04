import os
import tempfile
from PIL import Image

from onnxruntime_extensions.pp_api import ImageProcessor

img_proc = ImageProcessor(R"""
{
  "processor": {
    "name": "image_processing",
    "transforms": [
      {
        "operation": {
          "name": "decode_image",
          "type": "DecodeImage",
          "attrs": {
            "color_space": "BGR"
          }
        }
      },
      {
        "operation": {
          "name": "convert_to_rgb",
          "type": "ConvertRGB"
        }
      }
    ]
  }
}""")

img_name = "australia.jpg"
result = img_proc.pre_process(os.path.dirname(__file__) + "/" + img_name)
np_img = img_proc.to_numpy(result)
print(np_img.shape, np_img.dtype)

# can save the image back to disk
img_rgb = np_img[0]
img_bgr = img_rgb[..., ::-1]
output_name = tempfile.gettempdir() + "/" + img_name
Image.fromarray(img_bgr).save(output_name)
print(output_name)
