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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        # input should be uint8 data HWC
        input_dims = input_shape_str.split(",")
        assert input_type_str == "uint8" and len(input_dims) == 3 and input_dims[2] == "3"

        # https://en.wikipedia.org/wiki/YCbCr
        # exact weights from https://www.itu.int/rec/T-REC-T.871-201105-I/en
        rgb_weights = np.array([[0.299, 0.587, 0.114],
                                [-0.299 / 1.772, -0.587 / 1.772, 0.500],
                                [0.500, -0.587 / 1.402, -0.114 / 1.402]],
                               dtype=np.float32)  # fmt: skip

        bias = [0.0, 128.0, 128.0]

        if self._layout == "RGB":
            weights = rgb_weights
        else:
            weights = rgb_weights[:, ::-1]  # reverse the order of the last dim for BGR input

        # Weights are transposed for usage in matmul.
        weights_shape = "3, 3"
        weights = ",".join([str(w) for w in weights.T.flatten()])

        bias_shape = "3"
        bias = ",".join([str(b) for b in bias])

        # each output is {h, w}. TBD if input is CHW or HWC though. Once we figure that out we could copy values from
        # the input shape
        output_shape_str = f"YCbCr_ppp_{self.step_num}_h, YCbCr_ppp_{self.step_num}_w"
        assert input_type_str == "uint8"

        split_attr = "axis = -1"
        if onnx_opset >= 18:
            # Split now requires the number of outputs to be specified even though that can be easily inferred...
            split_attr += ", num_outputs = 3"

        # convert to float for MatMul
        # apply weights and bias
        # round and clip so it's in the range 0..255
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
                split_Y, split_Cb, split_Cr = Split <{split_attr}>(f_clipped)
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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
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
        # fmt: on

        # reverse first dim of weights for output to be bgr
        ycbcr_to_bgr_weights = ycbcr_to_rgb_weights[::-1, :]

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

    def __init__(self, resize_to: Union[int, Tuple[int, int]], layout: str = "HWC",
                 policy: str = "not_smaller", name: Optional[str] = None):
        """
        Args:
            resize_to: Target size. Can be a single value or a tuple with (target_height, target_width).
                       The aspect ratio will be maintained and neither height or width in the result will be smaller
                       than the requested value.
            layout: Input layout. 'NCHW', 'NHWC', 'CHW', 'HWC' and 'HW' are supported.
            policy: not_smaller (default) 
                        the sizes are adjusted so that no extent of the output is larger than the specified size, 
                        while keeping the original aspect ratio
                    not_larger
                        the sizes are adjusted so that no extent of the output is smaller than the specified size, 
                        while keeping the original aspect ratio.
                    Please refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize for more details.
            name: Optional name. Defaults to 'Resize'
        """
        super().__init__(["image"], ["resized_image"], name)
        if isinstance(resize_to, int):
            self._height = self._width = resize_to
        else:
            assert isinstance(resize_to, tuple)
            self._height, self._width = resize_to

        self._layout = layout
        self.policy_ = policy

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(",")

        # adjust for layout
        # resize will use the largest ratio so both sides won't necessarily match the requested height and width.
        # use symbolic names for the output dims as we have to provide values. prefix the names to try and
        # avoid any clashes.
        add_batch_dim = False

        if self._layout == "NHWC":
            assert len(dims) == 4
            split_str = "n, h, w, c"
            sizes_str = "n, h2, w2, c"
            output_shape_str = f"{dims[0]}, resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w, {dims[-1]}"
        elif self._layout == "NCHW":
            assert len(dims) == 4
            split_str = "n, c, h, w"
            sizes_str = "n, c, h2, w2"
            output_shape_str = f"{dims[0]}, {dims[1]}, resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w"
        elif self._layout == "HWC":
            assert len(dims) == 3
            add_batch_dim = True
            split_str = "h, w, c"
            sizes_str = "h2, w2, c"
            output_shape_str = f"resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w, {dims[-1]}"
        elif self._layout == "CHW":
            assert len(dims) == 3
            add_batch_dim = True
            split_str = "c, h, w"
            sizes_str = "c, h2, w2"
            output_shape_str = f"{dims[0]}, resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w"
        elif self._layout == "HW":
            assert len(dims) == 2
            split_str = "h, w"
            sizes_str = "h2, w2"
            output_shape_str = f"resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w"
        else:
            raise ValueError(f"Unsupported layout of {self._layout}")

        # TODO: Make this configurable. Matching PIL resize for now.
        resize_attributes = 'mode = "linear", nearest_mode = "floor"'
        if onnx_opset >= 18:
            # Resize matches PIL better if antialiasing is used, but that isn't available until ONNX opset 18.
            # Allow this to be used with older opsets as well.
            resize_attributes += ', antialias = 1'

        u64_1_str = ""

        # Rank 3 input uses trilinear interpolation, so if input is HWC or CHW we need to add a temporary batch dim
        # to make it rank 4, which will result in Resize using the desired bilinear interpolation.
        if add_batch_dim:
            u64_1_str = "u64_1 = Constant <value = int64[1] {1}> ()"
            sizes_str = "u64_1, " + sizes_str
            resize_str = \
                f"""\
                axes = Constant <value = int64[1] {{{0}}}> ()
                unsqueezed = Unsqueeze ({self.input_names[0]}, axes)
                resized =  Resize <{resize_attributes}> (unsqueezed, , , sizes_resize)
                {self.output_names[0]} = Squeeze (resized, axes)
                """
        else:
            resize_str = \
                f"{self.output_names[0]} = Resize <{resize_attributes}> ({self.input_names[0]}, , , sizes_resize)"

        split_input_shape_attr = "axis = 0"
        split_new_sizes_attr = "axis = 0"
        if onnx_opset >= 18:
            # Split now requires the number of outputs to be specified even though that can be easily inferred...
            split_input_shape_attr += f", num_outputs = {len(dims)}"
            split_new_sizes_attr += ", num_outputs = 2"
        
        # Resize-18 has the attribute "not_larger/not_smaller" to specify the resize policy, however
        # we want to support older opsets as well. 
        assert self.policy_ in ["not_smaller", "not_larger"], \
            f"Unsupported resize policy of {self.policy_}, must be 'not_smaller' or 'not_larger'"

        ratio_resize_func = "ReduceMax"
        if self.policy_ == "not_larger":
            ratio_resize_func = "ReduceMin"

        resize_graph = onnx.parser.parse_graph(
            f"""\
            resize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) => 
                ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_size = Constant <value = float[2] {{{float(self._height)}, {float(self._width)}}}> ()
                image_shape = Shape ({self.input_names[0]})
                {split_str} = Split <{split_input_shape_attr}> (image_shape)
                hw = Concat <axis = 0> (h, w)
                f_hw = Cast <to = 1> (hw)
                ratios = Div (target_size, f_hw)
                ratio_resize = {ratio_resize_func} (ratios)
                f_hw2_exact = Mul (f_hw, ratio_resize)
                f_hw2_round = Round (f_hw2_exact)
                hw2 = Cast <to = 7> (f_hw2_round)
                h2, w2 = Split <{split_new_sizes_attr}> (hw2)
                {u64_1_str}
                sizes_resize = Concat <axis = 0> ({sizes_str})
                {resize_str}
            }}
            """
        )

        return resize_graph


class CenterCrop(Step):
    """
    Crop the input to the requested dimensions, with the crop being centered.
    Currently only HWC input is handled.
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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
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
                hw = Gather (x_shape, axes)
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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
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

    def __init__(self, rescale_factor: float = 1/255, name: Optional[str] = None):
        """
        Args:
            name: Optional step name. Defaults to 'ImageBytesToFloat'
        """
        super().__init__(["data"], ["float_data"], name)
        self.rescale_factor_ = rescale_factor

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
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
                f_scale = Constant <value = float[1] {{{self.rescale_factor_}}}>()

                {optional_cast}
                {self.output_names[0]} = Mul(input_f, f_scale)
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

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert input_type_str == "float"

        if self._multiplier == 1.0:
            scale_input = ''
            scaled_input_name = self.input_names[0]
        else:
            scale_input = \
                f"""\
                f_multiplier = Constant <value = float[1] {{{self._multiplier}}}> ()
                scaled_input = Mul ({self.input_names[0]}, f_multiplier)
                """
            scaled_input_name = 'scaled_input'

        float_to_byte_graphs = onnx.parser.parse_graph(
            f"""\
            float_to_type (float[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{input_shape_str}] {self.output_names[0]})
            {{
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}>()
                
                {scale_input}
                rounded = Round ({scaled_input_name})
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


class DrawBoundingBoxes(Step):
    """
    Draw boxes on BGR image at given position, image is channel last and ordered by BGR.
    Input shape: <uint8_t>{height, width, 3<BGR>}
    boxes: <float>{num_boxes, 6<x, y, x/w, y/h, score, class>}
        The coordinates is the absolute pixel values in the picture. Its value is determined by `mode`.
        we have different modes to represent the coordinates of the box.[XYXY, XYWH, CENTER_XYWH].
        Please refer to the following link for more details. https://keras.io/api/keras_cv/bounding_box/formats/
        **score** is the confidence of the box(object score * class probability) and **class** is the class of the box.

    Output shape: <uint8_t>{height, width, 3<BGR>}
    """

    def __init__(self, mode: str = "XYXY", thickness: int = 4, num_classes: int = 10,
                 colour_by_classes=False, name: Optional[str] = None):
        """
        Args:
            mode: The mode of the boxes, 
                    "XYXY" (xmin ymin xmax ymax)  All values in the XYXY format should be absolute pixel values.
                    "XYWH" (xmin ymin width height) 
                    "CENTER_XYWH" (x_center, y_center, width, height) 
                                  All values in the CENTER_XYWH format should be absolute pixel values.


            thickness: Thickness of the box edge
            num_colours: Number of colours to use
                         We support 10 predefined colours and the other classes more than 10 wouldn't be drawn.
                         colors are [Red, Yellow, Lime, Cyan, Blue, Magenta, Orange, Maroon, Green, Navy]
                         and are used in that order. i.e. result with best score will use red. 
            colour_by_classes: Colour boxes by classes or by score. 
                               If `True` we use a colour for each unique class, with all results from the top 
                               `num_colours` classes displayed. A colour is only used for a single class. 
                               If `False`, we draw boxes for the top `num_colours` results. A colour is used 
                               for a single result, regardless of class.
            name: Optional name of step. Defaults to 'DrawBoundingBoxes'
        """
        super().__init__(["image", "boxes"], ["image_out"], name)
        self.thickness_ = thickness
        self.num_classes_ = num_classes
        self.colour_by_classes_ = colour_by_classes
        self.mode_ = mode

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input0_type_str, input0_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        input1_type_str, input1_shape_str = self._get_input_type_and_shape_strs(graph, 1)
        assert input0_type_str == "uint8" and input1_type_str == "float"

        assert str(input1_shape_str.split(",")[-1]) == "6"


        output_shape_str = input0_shape_str
        converter_graph = onnx.parser.parse_graph(
            f"""\
            bounding_box (uint8[{input0_shape_str}] {self.input_names[0]}, float[{input1_shape_str}] {self.input_names[1]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = com.microsoft.extensions.DrawBoundingBoxes({self.input_names[0]}, {self.input_names[1]})
            }}
            """
        )
        op_attr = ["thickness", "num_classes", "colour_by_classes","mode"]
        token_model_attr = []
        token_model_attr.append(onnx.helper.make_attribute(op_attr[0], self.thickness_))
        token_model_attr.append(onnx.helper.make_attribute(op_attr[1], self.num_classes_))
        token_model_attr.append(onnx.helper.make_attribute(op_attr[2], int(self.colour_by_classes_)))
        token_model_attr.append(onnx.helper.make_attribute(op_attr[3], self.mode_))
        converter_graph.node[0].attribute.extend(token_model_attr)

        return converter_graph


class LetterBox(Step):
    """
    Image is channel last and ordered by BGR.
    mainly used in object detection, it mostly follows behind resize operation. 
    This step either add border or crop the image to satisfy network input.
    -----          bbbbbbbbb
    |img|    --- > bb-----bb  
    -----          bb|img|bb
                   bb-----bb
                   bbbbbbbbb
    If target_shape is less than the original image, it will crop the image in a center mode.
    And the padding values will be negative and the Pad op performs cropping.

    Input shape: <uint8_t>{height, width, 3<BGR>}
    target_shape: <uint8_t>{out_height, out_width, 3<BGR>}
    layout: HWC or CHW are supported
    Output shape: specified by target_shape
    """

    def __init__(self, target_shape: Union[int, Tuple[int, int]], fill_value=0, layout: str = "HWC",
                 name: Optional[str] = None):
        """
        Args:
            target_shape: the size of the output image
            fill_value:  a constant value used to fill the border
            name: Optional name of step. Defaults to 'LetterBox'
        """            
        super().__init__(["image"], ["image_pad"], name)

        self.target_shape_ = target_shape
        self.fill_value_ = fill_value

        if layout != "HWC" and layout != "CHW":
            raise ValueError("Invalid layout. Only HWC and CHW are supported")

        self.layout_ = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input0_type_str, input0_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert len(input0_shape_str.split(',')) == 3, "expected HWC or CHW input"

        target_shape = f"{self.target_shape_[0]}, {self.target_shape_[1]}"

        if self.layout_ == "HWC":
            target_shape_str = f"{target_shape}, 3"
            split_input_shape_output = "h, w, c"
            concat_input_order = "half_pad_hw, i64_0, remainder_pad_hw, i64_0"
        else:
            target_shape_str = f"3, {target_shape}"
            split_input_shape_output = "c, h, w"
            concat_input_order = "i64_0, half_pad_hw, i64_0, remainder_pad_hw"

        split_input_shape_attr = "axis = 0"
        if onnx_opset >= 18:
            # Split now requires the number of outputs to be specified even though that can be easily inferred...
            split_input_shape_attr += f", num_outputs = 3"

        graph_text = (
            f"""\
            LetterBox (uint8[{input0_shape_str}] {self.input_names[0]}) 
                => (uint8[{target_shape_str}] {self.output_names[0]})  
            {{
                target_size = Constant <value = int64[2] {{{(self.target_shape_[0])}, {(self.target_shape_[1])}}}> ()
                i64_2 = Constant <value = int64[1] {{2}}>()
                i64_0 = Constant <value = int64[1] {{0}}>()
                const_val = Constant <value = uint8[1] {{{self.fill_value_}}}> ()
                image_shape = Shape ({self.input_names[0]})
                {split_input_shape_output} = Split <{split_input_shape_attr}> (image_shape)
                hw = Concat <axis = 0> (h, w)
                pad_hw = Sub (target_size, hw)
                half_pad_hw = Div (pad_hw, i64_2)
                remainder_pad_hw = Sub (pad_hw, half_pad_hw)
                pad_value = Concat <axis = 0> ({concat_input_order})
                {self.output_names[0]} = Pad({self.input_names[0]}, pad_value, const_val)
            }}
            """
        )

        converter_graph = onnx.parser.parse_graph(graph_text)

        return converter_graph


class SplitOutBoxAndScoreWithConf(Step):
    r"""
    Split the output of the model into boxes and scores, applying the object confidence score.
    Input shape: <float>{num_boxes, <4 box co-ords, conf score, num_classes>}
    Output shape: <float>{num_boxes, 4}, <float>{num_boxes, num_classes}
    |x1,x2,x3,x4, obj_conf, cls_1, ... cls_num|
            /\
           /  \
    |x1,x2,x3,x4|  |cls_1, ... clx_num|*obj_conf
    """

    def __init__(self, num_classes: int, name: Optional[str] = None):
        """
        Args:
            num_classes: number of classes
            name: Optional name of step. Defaults to 'SplitOutBoxAndScoreWithConf'
        """

        super().__init__(["box_conf_scores"], ["boxes", "scores"], name)
        self.num_classes_ = num_classes

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input0_type_str, input0_shape_str = self._get_input_type_and_shape_strs(graph, 0)

        input_shape_list = input0_shape_str.split(',')
        assert len(input_shape_list) == 2, " expected [num_boxes, 5+num_classes]"

        target_shape_str_0 = f"{input_shape_list[0]}, 4"
        target_shape_str_1 = f"{input_shape_list[0]}, _{self._step_num}_class"

        converter_graph = onnx.parser.parse_graph(
            f"""\
            SplitOutBoxConfidenceAndScore (float[{input0_shape_str}] {self.input_names[0]}) 
                => (float[{target_shape_str_0}] {self.output_names[0]}, 
                    float[{target_shape_str_1}] {self.output_names[1]})
            {{
                split_sizes = Constant <value = int64[3] {{4, 1, {self.num_classes_}}}>()
                {self.output_names[0]}, conf, orig_scores = Split <axis=-1>({self.input_names[0]}, split_sizes)

                scores_with_conf = Mul(orig_scores, conf)
                {self.output_names[1]} = Identity (scores_with_conf)
            }}
            """
        )
        return converter_graph


class SelectBestBoundingBoxesByNMS(Step):
    """
    Non-maximum suppression (NMS) is to select the best bounding boxes.
    Input:
        boxes: float[num_boxes, 4]
        scores: float[num_boxes, num_classes]
        masks: float[num_boxes, mask_data]. optional

    Output:
        nms_out: float[_few_num_boxes, <box+score+class+mask_data>]
    """

    def __init__(self,
                 iou_threshold: Optional[float] = 0.5,
                 score_threshold: Optional[float] = 0.67,
                 max_boxes_per_class: Optional[int] = 100,
                 max_detections: Optional[int] = None,
                 has_mask_data: Optional[bool] = False, name: Optional[str] = None):
        """
        Args: Please refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
              for more details about the parameters.
            iou_threshold: same as NonMaxSuppression op, intersection/union of boxes
            score_threshold: If this box's score is lower than score_threshold, it will be removed.
            max_boxes_per_class: max number of boxes to be selected per class
            max_detections: maximum number of boxes in total. Applied as the last step of processing if specified.
            name: Optional name of step. Defaults to 'SelectBestBoundingBoxesByNMS'
        """
        inputs = ["boxes", "scores"]
        if has_mask_data:
            inputs.append("masks")

        super().__init__(inputs, ["nms_out"], name)

        self.iou_threshold_ = iou_threshold
        self.score_threshold_ = score_threshold
        self.max_boxes_per_class_ = max_boxes_per_class
        self.max_detections_ = max_detections

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input0_type_str, input0_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        input1_type_str, input1_shape_str = self._get_input_type_and_shape_strs(graph, 1)

        input0_shape_list = input0_shape_str.split(',')
        assert len(input0_shape_list) == 2, " expected [num_boxes, 4]"

        has_mask_input = len(self.input_names) == 3

        input_2 = ""
        mask_i = ""
        mask_select = ""
        concat_for_output = "boxes_select, score_select, class_select"
        output_size_str = "6"
        # reduce_score picks the class with the best score for the selected box
        reduce_score = '(score_select_nm, i64_neg1)' if onnx_opset >= 18 else '<axes=[-1]>(score_select_nm)'

        if has_mask_input:
            input2_type_str, input2_shape_str = self._get_input_type_and_shape_strs(graph, 2)
            input_2 = f", float[{input2_shape_str}] {self.input_names[2]}"
            mask_i = f"masks_i = Identity({self.input_names[2]})"
            mask_select = "mask_select = Gather <axis=0>(masks_i, box_idxs)"
            concat_for_output += ", mask_select"

            mask_size_str = input2_shape_str.split(",")[-1]
            if mask_size_str.isnumeric():
                output_size_str = str(6 + int(mask_size_str))
            else:
                output_size_str = f"_step{self._step_num}_6_+_mask_size"

        if self.max_detections_:
            # squeeze scores from [num_results, 1] to [num_results]
            # use TopK to find the best scores for the selected boxes, but only if the number of results is
            # greater than max_detections, and there are results (otherwise calling TopK is invalid).
            # We sort the selected indices to maintain the original ordering for consistency when TopK isn't required
            apply_max_detections = \
                f"""
                max_detections = Constant <value = int64[1] {{{self.max_detections_}}}>()
                num_results = Shape(scores)
                num_results_less_than_max = Less(num_results, max_detections)
                k = Where(num_results_less_than_max, num_results, max_detections)
                have_results = Greater(k, i64_0)
                final_results = If<
                                then_branch=then_graph() => 
                                    (float[_{self._step_num}_selected_boxes, {output_size_str}] then_output) 
                                    {{
                                        topk_scores, topk_i = TopK<axis = 0>(scores, k)
                                        # use Unique to sort. no onnx op seems to provide that directly.
                                        sorted_topk_i = Unique<sorted=1>(topk_i)
                                        then_output = Gather<axis = 0>(merged_results, sorted_topk_i)
                                    }},
                                else_branch=else_graph() => 
                                    (float[_{self._step_num}_selected_boxes, {output_size_str}] else_output) 
                                    {{
                                        else_output = Identity(merged_results)
                                    }}>
                                    (have_results)
                """

        else:
            apply_max_detections = "final_results = Identity(merged_results)"

        graph_text = \
            f"""
            SelectBestBoundingBoxesByNMS (float[{input0_shape_str}] {self.input_names[0]},
                                          float[{input1_shape_str}] {self.input_names[1]}
                                          {input_2}) 
                => (float[_{self._step_num}_selected_boxes, {output_size_str}] {self.output_names[0]})  
            {{
                i64_neg1 = Constant <value = int64[1] {{-1}}>()
                i64_0 = Constant <value = int64[1] {{0}}>()
                i64_1 = Constant <value = int64[1] {{1}}>()
                i64_2 = Constant <value = int64[1] {{2}}>()
                i64_1_2 = Constant <value = int64[2] {{1, 2}}>()
                max_per_class = Constant <value = int64[1] {{{self.max_boxes_per_class_}}}>()
                iou_th = Constant <value = float[1] {{{self.iou_threshold_}}}>()
                score_th = Constant <value = float[1] {{{self.score_threshold_}}}>()

                boxes_i = Identity({self.input_names[0]})
                scores_i = Identity({self.input_names[1]})
                {mask_i}
                
                scores_c_b = Transpose<perm=[1,0]>(scores_i)
                batch_boxes = Unsqueeze(boxes_i, i64_0)
                batch_scores = Unsqueeze(scores_c_b, i64_0)

                # NMS returns [num_selected_boxes, 3] where each entry is [batch, class idx, box idx] 
                nmsbox = NonMaxSuppression<center_point_box=1>(batch_boxes, batch_scores, max_per_class,
                                                               iou_th, score_th)
                                                               
                # extract class values
                nms_classes = Gather<axis=-1>(nmsbox, i64_1)
                class_select = Cast<to = 1>(nms_classes)

                # extract box indexes and select box info using them.
                nms_boxes = Gather<axis=-1>(nmsbox, i64_2)
                box_idxs = Squeeze(nms_boxes, i64_neg1)
                boxes_select = Gather<axis=0>(boxes_i, box_idxs)

                # scores_c_b is [classes, boxes]
                # box_class_idxs is [selected_boxes, 2] where the 2 values are class idx, box idx
                class_box_idxs = Gather<axis=-1>(nmsbox, i64_1_2)
                scores = GatherND(scores_c_b, class_box_idxs)
                score_select = Unsqueeze(scores, i64_neg1)
                
                {mask_select}
                
                merged_results = Concat <axis = -1> ({concat_for_output})
                
                {apply_max_detections}
                
                {self.output_names[0]} = Identity(final_results)
            }}
            """

        converter_graph = onnx.parser.parse_graph(graph_text)

        return converter_graph


class ScaleNMSBoundingBoxesAndKeyPoints(Step):
    """
    Scale bounding box and key point coordinates in optional mask data to original image.

    Input image goes through Resize and LetterBox steps during pre-processing (in that order), and the output of this
    is what the original model runs against.
    To display the predictions on the original image we need to apply the reverse size changes to the co-ordinates 
    of the bounding boxes.

    nms_step_output inner dimension has 4 values for the bounding box, 1 for the score, 1 for the selected class,
    and the remainder (if any) is the mask data.

    The mask data has values for a fixed number of key points. Each key point has an x and y value, and optionally a
    confidence value.

    input:
        nms_step_output: output of SelectBestBoundingBoxesByNMS Step, shape [num_boxes, 6+]
        original_image: original image decoded from jpg/png, <uint8_t>[H, W, 3] or [3, H, W]
        resized_image: output from Resize pre-processing Step, <uint8_t>[H1, W1, 3] or [3, H1, W1]
        letter_boxed_image: output from LetterBox pre-processing Step, <uint8_t>[H2, W2, 3] or [3, H2, W2]
        num_key_points: number of key points in each mask data entry, if present. optional.
    
    output:
        nms_output_with_scaled_boxes_and_keypoints: input data with boxes and key points scaled to original image.
    """

    def __init__(self, num_key_points: Optional[int] = 0, layout: Optional[str] = "HWC", name: Optional[str] = None):
        """
        Args:
            num_key_points: Number of key points in mask data. Only required if input has optional mask data.
            layout: HWC or CHW. Used to determine where to read the H and W value from the input image shapes.
                    MUST be the same for all 3 input images.

            name: Optional name of step. Defaults to 'ScaleNMSBoundingBoxesAndKeyPoints'
        """
        super().__init__(["nms_step_output", "original_image", "resized_image", "letter_boxed_image"],
                         ["nms_output_with_scaled_boxes_and_keypoints"], name)
        self._num_key_points = num_key_points

        if layout != "HWC" and layout != "CHW":
            raise ValueError("Invalid layout. Only HWC and CHW are supported")

        self.layout_ = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_params = []

        for idx, input_name in enumerate(self.input_names):
            input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, idx)
            graph_input_params.append(f"{input_type_str}[{input_shape_str}] {input_name}")

        graph_input_params = ', '.join(graph_input_params)

        if self.layout_ == "HWC":
            orig_image_h_w_c = "oh, ow, oc"
            scaled_image_h_w_c = "sh, sw, sc"
            letterboxed_image_h_w_c = "lh, lw, lc"
        else:
            orig_image_h_w_c = "oc, oh, ow"
            scaled_image_h_w_c = "sc, sh, sw"
            letterboxed_image_h_w_c = "lc, lh, lw"

        def split_num_outputs(num_outputs: int):
            split_input_shape_attr = ''
            if onnx_opset >= 18:
                split_input_shape_attr = f", num_outputs = {num_outputs}"
            return split_input_shape_attr

        nms_output_type_str, nms_output_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        nms_output_shape = nms_output_shape_str.split(',')
        data_size_per_result = nms_output_shape[-1]
        if not data_size_per_result.isnumeric():
            # this should be known when adding pre-processing
            raise ValueError("Shape of input must have numeric value for the mask data size")

        data_num_splits = 3  # splits of nms data into box[:2], box[2:4] , score+class, [mask]
        data_split_sizes = "2, 2, 2"  # sizes of the splits
        score_class_masks = "score_class"  # output name/s for trailing output/s from Split
        keypoint_processing = ""  # operators to process the keypoints
        scaled_keypoints = ""  # optional output from keypoint scaling

        data_size = int(data_size_per_result)
        if data_size > 6:
            # we have mask data to split out
            data_num_splits = 4
            keypoint_data_size = data_size - 6
            data_split_sizes += f", {keypoint_data_size}"
            score_class_masks = "score_class, masks"
            scaled_keypoints = ", scaled_keypoints"

            values_per_keypoint = int(keypoint_data_size / self._num_key_points)
            reshape_keypoints_to = ",".join([str(self._num_key_points), str(values_per_keypoint)])

            if keypoint_data_size > 2:
                # split into xy and conf
                keypoints_xy_and_conf_from_keypoints = \
                    f"""
                    keypoints_split_sizes = Constant <value = int64[2] {{2, {values_per_keypoint - 2}}}>()
                    keypoints_xy, conf = Split <axis = -1>(keypoints, keypoints_split_sizes)
                    """
                # need to re-combine after scaling
                scaled_keypoints_and_conf = "scaled_keypoints_and_conf = Concat <axis=-1>(scaled_keypoints_xy, conf)"

            else:
                # use the keypoint data as-is as we don't have 'conf' data to split out
                keypoints_xy_and_conf_from_keypoints = "keypoints_xy = Identity(keypoints)"
                scaled_keypoints_and_conf = "scaled_keypoints_and_conf = Identity(scaled_keypoints_xy)"

            keypoint_processing = \
                f"""
                reshape_keypoints_to = Constant <value = int64[2] {{{reshape_keypoints_to}}}>()
                input_shape = Shape ({self.input_names[0]})

                i64_0 = Constant <value = int64[1] {{0}}>()
                num_boxes = Gather <axis=0>(input_shape, i64_0)
                reshape_masks_to = Concat<axis=-1> (num_boxes, reshape_keypoints_to)
                keypoints = Reshape(masks, reshape_masks_to)
                
                {keypoints_xy_and_conf_from_keypoints}
                
                offset_keypoints_xy = Sub (keypoints_xy, f_half_pad_wh)
                scaled_keypoints_xy = Mul (offset_keypoints_xy, ratios)
                
                {scaled_keypoints_and_conf}
                
                orig_shape = Shape(masks)
                scaled_keypoints = Reshape(scaled_keypoints_and_conf, orig_shape)
                """

        graph_text = \
            f"""\
            ScaleNMSBoundingBoxesAndKeyPoints 
            ({graph_input_params}) => ({nms_output_type_str}[{nms_output_shape_str}] {self.output_names[0]})
            {{
                i64_2 = Constant <value = int64[1] {{2}}>()
                data_split_sizes = Constant <value = int64[{data_num_splits}] {{{data_split_sizes}}}>()
                
                boxes_xy, boxes_wh_or_xy, {score_class_masks} = Split <axis=-1>({self.input_names[0]}, data_split_sizes)
                    
                ori_shape = Shape ({self.input_names[1]})
                scaled_shape = Shape ({self.input_names[2]})
                lettered_shape = Shape ({self.input_names[3]})
                {orig_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (ori_shape)
                {scaled_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (scaled_shape)
                {letterboxed_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (lettered_shape)
                swh = Concat <axis = -1> (sw,sh)
                lwh = Concat <axis = -1> (lw,lh)
                
                f_oh = Cast <to = 1> (oh)
                f_sh = Cast <to = 1> (sh)
                ratios = Div (f_oh, f_sh)
                
                pad_wh = Sub (lwh, swh)
                half_pad_wh = Div (pad_wh, i64_2)
                f_half_pad_wh = Cast <to = 1> (half_pad_wh)

                offset_boxes_xy = Sub (boxes_xy, f_half_pad_wh)
                restored_boxes = Concat <axis=-1> (offset_boxes_xy, boxes_wh_or_xy)
                scaled_boxes = Mul (restored_boxes, ratios)
                
                {keypoint_processing}
                
                {self.output_names[0]} = Concat <axis=-1> (scaled_boxes, score_class {scaled_keypoints})
            }}
            """

        converter_graph = onnx.parser.parse_graph(graph_text)

        return converter_graph
