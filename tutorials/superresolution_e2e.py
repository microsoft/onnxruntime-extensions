import io
import numpy as np
import os
import sys

from pathlib import Path
from PIL import Image

_this_dirpath = Path(os.path.dirname(os.path.abspath(__file__)))
_ppp_script_dirpath = _this_dirpath / '..' / 'tools'

# add the path with the scripts to sys.path so we can import them
sys.path.append(str(_ppp_script_dirpath.resolve()))

ONNX_MODEL = 'pytorch_superresolution.onnx'
ONNX_MODEL_WITH_PRE_POST_PROCESSING = 'pytorch_superresolution.with_pre_post_processing.onnx'


# Export pytorch superresolution model as per
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
def convert_pytorch_superresolution_to_onnx():
    import torch.utils.model_zoo as model_zoo
    import torch.onnx

    # Super Resolution model definition in PyTorch
    import torch.nn as nn
    import torch.nn.init as init

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
            x = self.pixel_shuffle(self.conv4(x))
            return x

        def _initialize_weights(self):
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)

    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)

    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1  # fix batch size to 1 for use in mobile scenarios

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()

    # Create random input to the model and run it
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      ONNX_MODEL,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=15,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names


def add_pre_post_processing(output_format: str = "png"):
    # expected usage would be to run tools/add_pre_post_processing_to_model.py.
    # `python ./tools/add_pre_post_processing_to_model.py --help`
    #
    # we import the super resolution helper func directly do it programmatically here for convenience
    import add_pre_post_processing_to_model as add_ppp
    # add the processing to the model and output a PNG format image. JPG is also valid.
    add_ppp.superresolution(Path(ONNX_MODEL), Path(ONNX_MODEL_WITH_PRE_POST_PROCESSING), output_format)


def run_updated_onnx_model():
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    so = ort.SessionOptions()
    # register the custom operators for the image decode/encode pre/post processing  provided by onnxruntime-extensions
    # with onnxruntime. if we do not do this we'll get an error on model load about the operators not being found.
    ortext_lib_path = get_library_path()
    so.register_custom_ops_library(ortext_lib_path)
    inference_session = ort.InferenceSession(ONNX_MODEL_WITH_PRE_POST_PROCESSING, so)

    #
    test_image_path = _this_dirpath / 'data' / 'super_res_input.png'
    test_image_bytes = np.fromfile(test_image_path, dtype=np.uint8)
    outputs = inference_session.run(['image_out'], {'image': test_image_bytes})
    upsized_image_bytes = outputs[0]

    original = Image.open(io.BytesIO(test_image_bytes))
    updated = Image.open(io.BytesIO(upsized_image_bytes))

    return original, updated


if __name__ == '__main__':
    convert_pytorch_superresolution_to_onnx()
    add_pre_post_processing('png')
    original_img, updated_img = run_updated_onnx_model()
    new_width, new_height = updated_img.size

    # create a side-by-side image with both.
    # do a plain resize of original so side-by-side is an easier comparison
    resized_orig_img = original_img.resize((new_width, new_height))
    combined = Image.new('RGB', (new_width * 2, new_height))

    combined.paste(resized_orig_img, (0, 0))
    combined.paste(updated_img, (new_width, 0))

    # NOTE: The output is significantly better with ONNX opset 18 as Resize supports anti-aliasing.
    combined.show('Original resized vs original vs Super Resolution resized')
