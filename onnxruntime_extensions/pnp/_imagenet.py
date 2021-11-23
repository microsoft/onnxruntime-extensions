import io
import onnx
import torch
from torch.nn.functional import interpolate
from torch.onnx import TrainingMode, export as _export
from ._base import ProcessingModule
from ._functions import onnx_where, onnx_greater


def _resize_param(img, size):
    y, x = tuple(img.shape[-2:])
    scale_y = size / y
    scale_x = size / x
    return onnx_where(onnx_greater(scale_x, scale_y), scale_x, scale_y)


class ImagenetPreProcessingLite(ProcessingModule):
    def __init__(self, size):
        super(ImagenetPreProcessingLite, self).__init__()
        self.target_size = size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))
        x = img.to(torch.float32).unsqueeze(0)
        # T.CenterCrop(224),
        width, height = tuple(self.target_size)
        img_h, img_w = x.shape[-2:]
        s_h = torch.div((img_h - height), 2, rounding_mode='trunc')
        s_w = torch.div((img_w - width), 2, rounding_mode='trunc')
        x = x[:, :, s_h:s_h + height, s_w:s_w + width]
        # T.ToTensor(),
        x /= 255.  # ToTensor
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        x -= torch.reshape(torch.tensor(mean), (3, 1, 1))
        x /= torch.reshape(torch.tensor(std), (3, 1, 1))
        return x


class ImagenetPreProcessing(ProcessingModule):
    def __init__(self, size):
        super(ImagenetPreProcessing, self).__init__()
        self.target_size = size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))
        # T.Resize(256),
        img = img.to(torch.float32).unsqueeze(0)
        scale = _resize_param(img, torch.tensor(256))
        x = interpolate(img, scale_factor=scale,
                        recompute_scale_factor=True,
                        mode="bilinear", align_corners=False)
        # T.CenterCrop(224),
        width, height = self.target_size, self.target_size
        img_h, img_w = x.shape[-2:]
        s_h = torch.div((img_h - height), 2, rounding_mode='trunc')
        s_w = torch.div((img_w - width), 2, rounding_mode='trunc')
        x = x[:, :, s_h:s_h + height, s_w:s_w + width]
        # T.ToTensor(),
        x /= 255.  # ToTensor
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x -= torch.reshape(torch.tensor(mean), (3, 1, 1))
        x /= torch.reshape(torch.tensor(std), (3, 1, 1))
        # x[:, 0, :, :] -= mean[0]
        # x[:, 1, :, :] -= mean[1]
        # x[:, 2, :, :] -= mean[2]
        # x[:, 0, :, :] /= std[0]
        # x[:, 1, :, :] /= std[1]
        # x[:, 2, :, :] /= std[2]
        return x


class ImagePostProcessing(ProcessingModule):
    def forward(self, scores):
        ProcessingModule.register_customops()
        probabilities = torch.softmax(scores, dim=1)
        ids = probabilities.argsort(dim=1, descending=True)
        return ids, probabilities

    def export(self, opset_version, *args):
        with io.BytesIO() as f:
            name_i = 'image'
            _export(self, args, f,
                    training=TrainingMode.EVAL,
                    opset_version=opset_version,
                    input_names=[name_i],
                    dynamic_axes={name_i: [0, 1]})
            return onnx.load_model(io.BytesIO(f.getvalue()))


class PreMobileNet(ImagenetPreProcessing):
    def __init__(self, size=None):
        super(PreMobileNet, self).__init__(224 if size is None else size)


class PostMobileNet(ImagePostProcessing):
    pass
