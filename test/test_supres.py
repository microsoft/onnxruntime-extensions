import io
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import unittest
from PIL import Image
from onnxruntime_extensions import OrtPyFunction, util


_input_image_filepath = util.get_test_data_file("data", "test_supres.jpg")
_onnx_model_filepath = util.get_test_data_file("data", "supres.onnx")
_torch_model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'


class SuperResolutionNet(nn.Module):
  def __init__(self, upscale_factor, inplace=False):
    super(SuperResolutionNet, self).__init__()

    self.relu = nn.ReLU(inplace=inplace)
    self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
    self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
    self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
    self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    self._initialize_weights()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))

    return self.pixel_shuffle(self.conv4(x))

  def _initialize_weights(self):
    nn.init.orthogonal_(self.conv1.weight, nn.init.calculate_gain('relu'))
    nn.init.orthogonal_(self.conv2.weight, nn.init.calculate_gain('relu'))
    nn.init.orthogonal_(self.conv3.weight, nn.init.calculate_gain('relu'))
    nn.init.orthogonal_(self.conv4.weight)


def _run_torch_inferencing():
  # Create the super-resolution model by using the above model definition.
  torch_model = SuperResolutionNet(upscale_factor=3)

  # Initialize & load model with the pretrained weights
  map_location = lambda storage, loc: storage
  if torch.cuda.is_available():
    map_location = None
  torch_model.load_state_dict(model_zoo.load_url(_torch_model_url, map_location=map_location))

  # set the model to inferencing mode
  torch_model.eval()

  input_image_ycbcr = Image.open(_input_image_filepath).convert('YCbCr')
  input_image_y, input_image_cb, input_image_cr = input_image_ycbcr.split()
  input_image_y = torch.from_numpy(np.asarray(input_image_y, dtype=np.uint8)).float()
  input_image_y /= 255.0
  input_image_y = input_image_y.view(1, -1, input_image_y.shape[1], input_image_y.shape[0])

  output_image_y = torch_model(input_image_y)
  output_image_y = output_image_y.detach().cpu().numpy()
  output_image_y = Image.fromarray(np.uint8((output_image_y[0] * 255.0).clip(0, 255)[0]), mode='L')

  # get the output image follow post-processing step from PyTorch implementation
  output_image = Image.merge(
      "YCbCr", [
          output_image_y,
          input_image_cb.resize(output_image_y.size, Image.BICUBIC),
          input_image_cr.resize(output_image_y.size, Image.BICUBIC),
      ]).convert("RGB")

  # Uncomment to create a local file
  #
  # output_image_filepath = util.get_test_data_file("data", "test_supres_torch.jpg")
  # output_image.save(output_image_filepath)

  output_image = np.asarray(output_image, dtype=np.uint8)
  return output_image


def _run_onnx_inferencing():
  encoded_input_image = open(_input_image_filepath, 'rb').read()
  encoded_input_image = np.frombuffer(encoded_input_image, dtype=np.uint8)

  onnx_model = OrtPyFunction.from_model(_onnx_model_filepath)
  encoded_output_image = onnx_model(encoded_input_image)

  encoded_output_image = encoded_output_image.tobytes()

  # Uncomment to create a local file
  #
  # output_image_filepath = util.get_test_data_file("data", "test_supres_onnx.jpg")
  # with open(output_image_filepath, 'wb') as strm:
  #   strm.write(encoded_output_image)
  #   strm.flush()

  with io.BytesIO(encoded_output_image) as strm:
    decoded_output_image = Image.open(strm).convert('RGB')
  
  decoded_output_image = np.asarray(decoded_output_image, dtype=np.uint8)
  return decoded_output_image


class TestSupres(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
      pass

  def test_e2e(self):
    actual = _run_onnx_inferencing()
    expected = _run_torch_inferencing()

    self.assertEqual(actual.shape[0], expected.shape[0])
    self.assertEqual(actual.shape[1], expected.shape[1])
    self.assertEqual(actual.shape[2], expected.shape[2])

    self.assertTrue(np.allclose(actual, expected, atol=20))


if __name__ == "__main__":
    unittest.main()
