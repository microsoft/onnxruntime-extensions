Module pre_post_processing.steps.vision
=======================================

Classes
-------

`CenterCrop(height: int, width: int, name: Optional[str] = None)`
:   Crop the input to the requested dimensions, with the crop being centered.
    Currently only HWC input is handled.
    
    Args:
        height: Height of area to crop.
        width: Width of area to crop.
        name: Optional step name. Defaults to 'CenterCrop'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`ChannelsLastToChannelsFirst(has_batch_dim: bool = False, name: Optional[str] = None)`
:   Convert channels last data to channels first.
    Input can be NHWC or HWC.
    
    Args:
        has_batch_dim: Set to True if the input has a batch dimension (i.e. is NHWC)
        name: Optional step name. Defaults to 'ChannelsLastToChannelsFirst'

    ### Ancestors (in MRO)

    * pre_post_processing.steps.general.Transpose
    * pre_post_processing.step.Step

`ConvertBGRToImage(image_format: str = 'jpg', name: Optional[str] = None)`
:   Convert BGR ordered uint8 data into an encoded image.
    Supported output input formats: jpg, png
    Input shape: {input_image_height, input_image_width, 3}
    Output shape: {num_encoded_bytes}
    
    Args:
        image_format: Format to encode to. jpg and png are supported.
        name: Optional step name. Defaults to 'ConvertBGRToImage'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`ConvertImageToBGR(name: Optional[str] = None)`
:   Convert the bytes of an image by decoding to BGR ordered uint8 values.
    Supported input formats: jpg, png
    Input shape: {num_encoded_bytes}
    Output shape: {input_image_height, input_image_width, 3}
    
    Args:
        name: Optional name of step. Defaults to 'ConvertImageToBGR'
    
    NOTE: Input image format is inferred and does not need to be specified.

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`FloatToImageBytes(multiplier: float = 255.0, name: Optional[str] = None)`
:   Converting floating point values to uint8 values in range 0..255.
    Typically this reverses ImageBytesToFloat by converting input data in the range 0..1, but an optional multiplier
    can be specified if the input data has a different range.
    Values will be rounded prior to clipping and conversion to uint8.
    
    Args:
        multiplier: Optional multiplier. Currently, the expected values are 255 (input data is in range 0..1), or
                    1 (input data is in range 0..255).
        name: Optional step name. Defaults to 'FloatToImageBytes'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`ImageBytesToFloat(name: Optional[str] = None)`
:   Convert uint8 or float values in range 0..255 to floating point values in range 0..1
    
    Args:
        name: Optional step name. Defaults to 'ImageBytesToFloat'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Normalize(normalization_values: List[Tuple[float, float]], layout: str = 'CHW', name: Optional[str] = None)`
:   Normalize input data on a per-channel basis.
        `x -> (x - mean) / stddev`
    Output is float with same shape as input.
    
    Args:
        normalization_values: Tuple with (mean, stddev). One entry per channel.
                              If single entry is provided it will be used for all channels.
        layout: Input layout. Can be 'CHW' or 'HWC'
        name: Optional step name. Defaults to 'Normalize'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`PixelsToYCbCr(layout: str = 'BGR', name: Optional[str] = None)`
:   Convert RGB or BGR pixel data to YCbCr format.
    Input shape: {height, width, 3}
    Output shape is the same.
    Output data is float, but rounded and clipped to the range 0..255 as per the spec for YCbCr conversion.
    
    Args:
        layout: Input data layout. Can be 'BGR' or 'RGB'
        name: Optional step name. Defaults to 'PixelsToYCbCr'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Resize(resize_to: Union[int, Tuple[int, int]], layout: str = 'HWC', name: Optional[str] = None)`
:   Resize input data. Aspect ratio is maintained.
    e.g. if image is 1200 x 600 and 300 x 300 is requested the result will be 600 x 300
    
    Args:
        resize_to: Target size. Can be a single value or a tuple with (target_height, target_width).
                   The aspect ratio will be maintained and neither height or width in the result will be smaller
                   than the requested value.
        layout: Input layout. 'NCHW', 'NHWC', 'CHW', 'HWC' and 'HW' are supported.
        name: Optional name. Defaults to 'Resize'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`YCbCrToPixels(layout: str = 'BGR', name: Optional[str] = None)`
:   Convert YCbCr input to RGB or BGR.
    
    Input data can be uint8 or float but all inputs must use the same type.
    Input shape: {height, width, 3}
    Output shape is the same.
    
    Args:
        layout: Output layout. Can be 'BGR' or 'RGB'
        name: Optional step name. Defaults to 'YCbCrToPixels'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step