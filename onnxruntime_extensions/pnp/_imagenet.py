import torch
from typing import Tuple
from torch.nn.functional import interpolate
from ._base import ProcessingTracedModule
from ._torchext import onnx_where, onnx_greater


def _resize_param(img, size):
    y, x = tuple(img.shape[-2:])
    scale_y = size / y
    scale_x = size / x
    return onnx_where(onnx_greater(scale_x, scale_y), scale_x, scale_y)


class ImageNetPreProcessing(ProcessingTracedModule):
    def __init__(self, size, resize_image=True):
        super(ImageNetPreProcessing, self).__init__()
        self.target_size = size
        self.resize_image = resize_image

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        assert img.shape[-1] == 3, 'the input image should be in RGB channels'
        img = torch.permute(img, (2, 0, 1))
        img = img.to(torch.float32).unsqueeze(0)
        # T.Resize(256),
        if self.resize_image:
            scale = _resize_param(img, torch.tensor(256))
            img = interpolate(img, scale_factor=scale,
                              recompute_scale_factor=True,
                              mode="bilinear", align_corners=False)
        # T.CenterCrop(224),
        width, height = self.target_size, self.target_size
        img_h, img_w = img.shape[-2:]
        s_h = torch.div((img_h - height), 2, rounding_mode='trunc')
        s_w = torch.div((img_w - width), 2, rounding_mode='trunc')
        x = img[:, :, s_h:s_h + height, s_w:s_w + width]
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
        return x


class ImageNetPostProcessing(ProcessingTracedModule):
    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(scores, dim=1)
        top10_prob, top10_ids = probabilities.topk(k=10, dim=1, largest=True, sorted=True)
        return top10_ids, top10_prob


class PreMobileNet(ImageNetPreProcessing):
    def __init__(self, size=None):
        super(PreMobileNet, self).__init__(224 if size is None else size)


class PostMobileNet(ImageNetPostProcessing):
    pass
