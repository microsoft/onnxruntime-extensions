# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
import numpy as np

from typing import List, Optional, Tuple, Union
from ..step import Step
from .general import Transpose


#
# Image conversion
#
class ConvertImageToBGR(Step):
    """
    Convert the bytes of an image by decoding to BGR ordered uint8 values.
    Supported input formats: jpg, png
    Input shape: {num_encoded_bytes}
    Output shape: {input_image_height, input_image_width, 3}
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Optional name of step. Defaults to 'ConvertImageToBGR'

        NOTE: Input image format is inferred and does not need to be specified.
        """
        super().__init__(["image"], ["bgr_data"], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert input_type_str == "uint8"
        output_shape_str = f"to_bgr_ppp_{self.step_num}_h, to_bgr_ppp_{self.step_num}_w, 3"

        converter_graph = onnx.parser.parse_graph(
            f"""\
            image_to_bgr (uint8[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} =  com.microsoft.extensions.DecodeImage({self.input_names[0]})
            }}
            """
        )

        return converter_graph


class ConvertBGRToImage(Step):
    """
    Convert BGR ordered uint8 data into an encoded image.
    Supported output input formats: jpg, png
    Input shape: {input_image_height, input_image_width, 3}
    Output shape: {num_encoded_bytes}
    """

    def __init__(self, image_format: str = "jpg", name: Optional[str] = None):
        """
        Args:
            image_format: Format to encode to. jpg and png are supported.
            name: Optional step name. Defaults to 'ConvertBGRToImage'
        """
        super().__init__(["bgr_data"], ["image"], name)
        assert image_format == "jpg" or image_format == "png"
        self._format = image_format

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert input_type_str == "uint8"
        output_shape_str = f"to_image_ppp_{self.step_num}_num_bytes"

        converter_graph = onnx.parser.parse_graph(
            f"""\
            bgr_to_image (uint8[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = com.microsoft.extensions.EncodeImage ({self.input_names[0]})
            }}
            """
        )

        # as this is a custom op we have to add the attribute for `format` directly to the node.
        # parse_graph doesn't have a schema for the operator and fails attempting to validate the attribute.
        format_attr = converter_graph.node[0].attribute.add()
        format_attr.name = "format"
        format_attr.type = onnx.AttributeProto.AttributeType.STRING
        format_attr.s = bytes(self._format, "utf-8")

        return converter_graph


class PixelsToYCbCr(Step):
    """
    Convert RGB or BGR pixel data to YCbCr format.
    Input shape: {height, width, 3}
    Output shape is the same.
    Output data is float, but rounded and clipped to the range 0..255 as per the spec for YCbCr conversion.
    """

    def __init__(self, layout: str = "BGR", name: Optional[str] = None):
        """
        Args:
            layout: Input data layout. Can be 'BGR' or 'RGB'
            name: Optional step name. Defaults to 'PixelsToYCbCr'
        """
        super().__init__(["pixels"], ["Y", "Cb", "Cr"], name)
        assert layout == "RGB" or layout == "BGR"
        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        # input should be uint8 data HWC
        input_dims = input_shape_str.split(",")
        assert input_type_str == "uint8" and len(input_dims) == 3 and input_dims[2] == "3"
        rgb_weights = np.array([[0.299, 0.587, 0.114],
                                [-0.299 / 1.772, -0.587 / 1.772, 0.500],
                                [0.500, -0.587 / 1.402, -0.114 / 1.402]],
                               dtype=np.float32)  # fmt: skip

        bias = [0.0, 128.0, 128.0]

        if self._layout == "RGB":
            weights = rgb_weights
        else:
            weights = rgb_weights[:, ::-1]  # reverse the order of the last dim to match

        # Weights are transposed for usage in matmul.
        weights_shape = "3, 3"
        weights = ",".join([str(w) for w in weights.T.flatten()])

        bias_shape = "3"
        bias = ",".join([str(b) for b in bias])

        # each output is {h, w}. TBD if input is CHW or HWC though. Once we figure that out we could copy values from
        # the input shape
        output_shape_str = f"YCbCr_ppp_{self.step_num}_h, YCbCr_ppp_{self.step_num}_w"
        assert input_type_str == "uint8"

        # convert to float for MatMul
        # apply weights and bias
        # round and clip so it's in the range 0..255
        # convert back to uint8
        # split into channels. shape will be {h, w, 1}
        # remove the trailing '1' so output is {h, w}
        converter_graph = onnx.parser.parse_graph(
            f"""\
            pixels_to_YCbCr (uint8[{input_shape_str}] {self.input_names[0]})
                => (float[{output_shape_str}] {self.output_names[0]},
                    float[{output_shape_str}] {self.output_names[1]},
                    float[{output_shape_str}] {self.output_names[2]})  
            {{
                kWeights = Constant <value = float[{weights_shape}] {{{weights}}}> ()
                kBias = Constant <value = float[{bias_shape}] {{{bias}}}> ()
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}> ()

                f_pixels = Cast <to = 1> ({self.input_names[0]})
                f_weighted = MatMul(f_pixels, kWeights)
                f_biased = Add(f_weighted, kBias)
                f_rounded = Round(f_biased)
                f_clipped = Clip (f_rounded, f_0, f_255)                                
                split_Y, split_Cb, split_Cr = Split <axis = -1>(f_clipped)
                {self.output_names[0]} = Squeeze (split_Y, i64_neg1)
                {self.output_names[1]} = Squeeze (split_Cb, i64_neg1)
                {self.output_names[2]} = Squeeze (split_Cr, i64_neg1)
            }}
            """
        )

        return converter_graph


class YCbCrToPixels(Step):
    """
    Convert YCbCr input to RGB or BGR.

    Input data can be uint8 or float but all inputs must use the same type.
    Input shape: {height, width, 3}
    Output shape is the same.
    """

    def __init__(self, layout: str = "BGR", name: Optional[str] = None):
        """
        Args:
            layout: Output layout. Can be 'BGR' or 'RGB'
            name: Optional step name. Defaults to 'YCbCrToPixels'
        """
        super().__init__(["Y", "Cb", "Cr"], ["bgr_data"], name)
        assert layout == "RGB" or layout == "BGR"
        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)
        input_type_str1, input_shape_str1 = self._get_input_type_and_shape_strs(graph, 1)
        input_type_str2, input_shape_str2 = self._get_input_type_and_shape_strs(graph, 2)
        assert (input_type_str0 == "uint8" and input_type_str1 == "uint8" and input_type_str2 == "uint8") or (
            input_type_str0 == "float" and input_type_str1 == "float" and input_type_str2 == "float"
        )

        assert (
            len(input_shape_str0.split(",")) == 2
            and len(input_shape_str1.split(",")) == 2
            and len(input_shape_str2.split(",")) == 2
        )

        output_shape_str = f"{input_shape_str0}, 3"

        # fmt: off
        # https://en.wikipedia.org/wiki/YCbCr
        # exact weights from https://www.itu.int/rec/T-REC-T.871-201105-I/en
        ycbcr_to_rgb_weights = np.array([[1, 0, 1.402],
                                         [1, -0.114*1.772/0.587, -0.299*1.402/0.587],
                                         [1, 1.772, 0]],
                                        dtype=np.float32)

        # reverse 2nd and 3rd entry in each row (YCbCr to YCrCb so blue and red are flipped)
        ycbcr_to_bgr_weights = np.array([[1, 1.402, 0],
                                         [1, -0.299*1.402/0.587, -0.114*1.772/0.587],
                                         [1, 0, 1.772]],
                                        dtype=np.float32)
        # fmt: on

        weights = ycbcr_to_bgr_weights if self._layout == "BGR" else ycbcr_to_rgb_weights
        bias = [0.0, 128.0, 128.0]

        weights_shape = "3, 3"
        # transpose weights for use in matmul
        weights = ",".join([str(w) for w in weights.T.flatten()])

        bias_shape = "3"
        bias = ",".join([str(b) for b in bias])

        # unsqueeze the {h, w} inputs to add channels dim. new shape is {h, w, 1}
        # merge Y, Cb, Cr data on the new channel axis
        # convert to float to apply weights etc.
        # remove bias
        # apply weights
        # round and clip to 0..255
        # convert to uint8.
        converter_graph = onnx.parser.parse_graph(
            f"""\
            YCbCr_to_RGB ({input_type_str0}[{input_shape_str0}] {self.input_names[0]},
                          {input_type_str1}[{input_shape_str1}] {self.input_names[1]},
                          {input_type_str2}[{input_shape_str2}] {self.input_names[2]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                kWeights = Constant <value = float[{weights_shape}] {{{weights}}}> ()
                kBias = Constant <value = float[{bias_shape}] {{{bias}}}> ()
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}> ()
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()

                Y1 = Unsqueeze({self.input_names[0]}, i64_neg1)
                Cb1 = Unsqueeze({self.input_names[1]}, i64_neg1)
                Cr1 = Unsqueeze({self.input_names[2]}, i64_neg1)
                YCbCr = Concat <axis = -1> (Y1, Cb1, Cr1)
                f_YCbCr = Cast <to = 1> (YCbCr)
                f_unbiased = Sub (f_YCbCr, kBias)
                f_pixels = MatMul (f_unbiased, kWeights)
                f_rounded = Round (f_pixels)
                clipped = Clip (f_rounded, f_0, f_255)
                {self.output_names[0]} = Cast <to = {onnx.TensorProto.UINT8}> (clipped)
            }}
            """
        )

        return converter_graph


#
# Pre-processing
#
class Resize(Step):
    """
    Resize input data. Aspect ratio is maintained.
    e.g. if image is 1200 x 600 and 300 x 300 is requested the result will be 600 x 300
    """

    def __init__(self, resize_to: Union[int, Tuple[int, int]], layout: str = "HWC", name: Optional[str] = None):
        """
        Args:
            resize_to: Target size. Can be a single value or a tuple with (target_height, target_width).
                       The aspect ratio will be maintained and neither height or width in the result will be smaller
                       than the requested value.
            layout: Input layout. 'CHW', 'HWC' and 'HW' are supported.
            name: Optional name. Defaults to 'Resize'
        """
        super().__init__(["image"], ["resized_image"], name)
        if isinstance(resize_to, int):
            self._height = self._width = resize_to
        else:
            assert isinstance(resize_to, tuple)
            self._height, self._width = resize_to

        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(",")

        # adjust for layout
        # resize will use the largest ratio so both sides won't necessarily match the requested height and width.
        # use symbolic names for the output dims as we have to provide values. prefix the names to try and
        # avoid any clashes
        scales_constant_str = "f_1 = Constant <value = float[1] {1.0}> ()"
        if self._layout == "HWC":
            assert len(dims) == 3
            split_str = "h, w, c"
            scales_str = "ratio_resize, ratio_resize, f_1"
            output_shape_str = f"resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w, {dims[-1]}"
        elif self._layout == "CHW":
            assert len(dims) == 3
            split_str = "c, h, w"
            scales_str = "f_1, ratio_resize, ratio_resize"
            output_shape_str = f"{dims[0]}, resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w"
        elif self._layout == "HW":
            assert len(dims) == 2
            split_str = "h, w"
            scales_str = "ratio_resize, ratio_resize"
            scales_constant_str = ""
            output_shape_str = f"resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w"
        else:
            raise ValueError(f"Unsupported layout of {self._layout}")

        # TODO: Make this configurable. Matching PIL resize for now
        resize_attributes = 'mode = "linear", nearest_mode = "floor"'

        resize_graph = onnx.parser.parse_graph(
            f"""\
            resize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) => 
                ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_size = Constant <value=float[2] {{{float(self._height)}, {float(self._width)}}}> ()
                image_shape = Shape ({self.input_names[0]})
                {split_str} = Split <axis=0> (image_shape)
                hw = Concat <axis = 0> (h, w)
                f_hw = Cast <to = 1> (hw)
                ratios = Div (target_size, f_hw)
                ratio_resize = ReduceMax (ratios)

                {scales_constant_str}
                scales_resize = Concat <axis = 0> ({scales_str})
                {self.output_names[0]} = Resize <{resize_attributes}> ({self.input_names[0]}, , scales_resize)
            }}
            """
        )

        return resize_graph


class CenterCrop(Step):
    """
    Crop the input to the requested dimensions, with the crop being centered.
    """

    def __init__(self, height: int, width: int, name: Optional[str] = None):
        """
        Args:
            height: Height of area to crop.
            width: Width of area to crop.
            name: Optional step name. Defaults to 'CenterCrop'
        """
        super().__init__(["image"], ["cropped_image"], name)
        self._height = height
        self._width = width

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(",")
        output_shape_str = f"{self._height}, {self._width}, {dims[-1]}"

        crop_graph = onnx.parser.parse_graph(
            f"""\
            crop ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_crop = Constant <value = int64[2] {{{self._height}, {self._width}}}> ()
                i64_2 = Constant <value = int64[1] {{2}}> ()
                axes = Constant <value = int64[2] {{0, 1}}> ()
                x_shape = Shape ({self.input_names[0]})
                h, w, c = Split <axis = 0> (x_shape)
                hw = Concat <axis = 0> (h, w)
                hw_diff = Sub (hw, target_crop)
                start_xy = Div (hw_diff, i64_2)
                end_xy = Add (start_xy, target_crop)
                {self.output_names[0]} = Slice ({self.input_names[0]}, start_xy, end_xy, axes)
            }}
            """
        )

        return crop_graph


class Normalize(Step):
    """
    Normalize input data on a per-channel basis.
        `x -> (x - mean) / stddev`
    Output is float with same shape as input.
    """

    def __init__(self, normalization_values: List[Tuple[float, float]], layout: str = "CHW", name: Optional[str] = None):
        """
        Args:
            normalization_values: Tuple with (mean, stddev). One entry per channel.
                                  If single entry is provided it will be used for all channels.
            layout: Input layout. Can be 'CHW' or 'HWC'
            name: Optional step name. Defaults to 'Normalize'
        """
        super().__init__(["data"], ["normalized_data"], name)

        # duplicate for each channel if needed
        if len(normalization_values) == 1:
            normalization_values *= 3

        assert len(normalization_values) == 3
        self._normalization_values = normalization_values
        assert layout == "HWC" or layout == "CHW"
        self._hwc_layout = True if layout == "HWC" else False

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        mean0 = self._normalization_values[0][0]
        mean1 = self._normalization_values[1][0]
        mean2 = self._normalization_values[2][0]
        stddev0 = self._normalization_values[0][1]
        stddev1 = self._normalization_values[1][1]
        stddev2 = self._normalization_values[2][1]

        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        values_shape = "3" if self._hwc_layout else "3, 1, 1"

        normalize_graph = onnx.parser.parse_graph(
            f"""\
            normalize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => (float[{input_shape_str}] {self.output_names[0]})
            {{
                kMean = Constant <value = float[{values_shape}] {{{mean0}, {mean1}, {mean2}}}> ()
                kStddev = Constant <value = float[{values_shape}] {{{stddev0}, {stddev1}, {stddev2}}}> ()
                f_input = Cast <to = 1> ({self.input_names[0]})
                f_sub_mean = Sub (f_input, kMean)
                {self.output_names[0]} = Div (f_sub_mean, kStddev)
            }}
            """
        )

        onnx.checker.check_graph(normalize_graph)
        return normalize_graph


#
# Utilities
#
class ImageBytesToFloat(Step):
    """
    Convert uint8 or float values in range 0..255 to floating point values in range 0..1
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Optional step name. Defaults to 'ImageBytesToFloat'
        """
        super().__init__(["data"], ["float_data"], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        if input_type_str == "uint8":
            optional_cast = f"""\
                input_f = Cast <to = 1> ({self.input_names[0]})
            """
        else:
            # no-op that optimizer will remove
            optional_cast = f"input_f = Identity ({self.input_names[0]})"

        byte_to_float_graph = onnx.parser.parse_graph(
            f"""\
            byte_to_float ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => (float[{input_shape_str}] {self.output_names[0]})
            {{
                f_255 = Constant <value = float[1] {{255.0}}>()

                {optional_cast}
                {self.output_names[0]} = Div(input_f, f_255)
            }}
            """
        )

        onnx.checker.check_graph(byte_to_float_graph)
        return byte_to_float_graph


class FloatToImageBytes(Step):
    """
    Converting floating point values to uint8 values in range 0..255.
    Typically this reverses ImageBytesToFloat by converting input data in the range 0..1, but an optional multiplier
    can be specified if the input data has a different range.
    Values will be rounded prior to clipping and conversion to uint8.
    """

    def __init__(self, multiplier: float = 255.0, name: Optional[str] = None):
        """
        Args:
            multiplier: Optional multiplier. Currently, the expected values are 255 (input data is in range 0..1), or
                        1 (input data is in range 0..255).
            name: Optional step name. Defaults to 'FloatToImageBytes'
        """
        super().__init__(["float_data"], ["pixel_data"], name)
        self._multiplier = multiplier

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert input_type_str == "float"

        float_to_byte_graphs = onnx.parser.parse_graph(
            f"""\
            float_to_type (float[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{input_shape_str}] {self.output_names[0]})
            {{
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}>()
                f_multiplier = Constant <value = float[1] {{{self._multiplier}}}> ()

                scaled_input = Mul ({self.input_names[0]}, f_multiplier)
                rounded = Round (scaled_input)
                clipped = Clip (rounded, f_0, f_255)
                {self.output_names[0]} = Cast <to = {onnx.TensorProto.UINT8}> (clipped)
            }}
            """
        )

        onnx.checker.check_graph(float_to_byte_graphs)
        return float_to_byte_graphs


class ChannelsLastToChannelsFirst(Transpose):
    """
    Convert channels last data to channels first.
    Input can be NHWC or HWC.
    """

    def __init__(self, has_batch_dim: bool = False, name: Optional[str] = None):
        """
        Args:
            has_batch_dim: Set to True if the input has a batch dimension (i.e. is NHWC)
            name: Optional step name. Defaults to 'ChannelsLastToChannelsFirst'
        """
        perms = [0, 3, 1, 2] if has_batch_dim else [2, 0, 1]
        super().__init__(perms, name)
