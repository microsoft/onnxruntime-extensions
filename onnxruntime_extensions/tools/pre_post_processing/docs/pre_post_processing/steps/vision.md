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

`DrawBoundingBoxes(mode: str = 'XYXY', thickness: int = 4, num_classes: int = 10, colour_by_classes=False, name: Optional[str] = None)`
:   Draw boxes on BGR image at given position, image is channel last and ordered by BGR.
    Input shape: <uint8_t>{height, width, 3<BGR>}
    boxes: <float>{num_boxes, 6<x, y, x/w, y/h, score, class>}
        The coordinates is the absolute pixel values in the picture. Its value is determined by `mode`.
        we have different modes to represent the coordinates of the box.[XYXY, XYWH, CENTER_XYWH].
        Please refer to the following link for more details. https://keras.io/api/keras_cv/bounding_box/formats/
        **score** is the confidence of the box(object score * class probability) and **class** is the class of the box.
    
    Output shape: <uint8_t>{height, width, 3<BGR>}
    
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

`LetterBox(target_shape: Union[int, Tuple[int, int]], fill_value=0, name: Optional[str] = None)`
:   Image is channel last and ordered by BGR.
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
    Output shape: specified by target_shape
    
    Args:
        target_shape: the size of the output image
        fill_value:  a constant value used to fill the border
        name: Optional name of step. Defaults to 'LetterBox'

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

`Resize(resize_to: Union[int, Tuple[int, int]], layout: str = 'HWC', policy: str = 'not_smaller', name: Optional[str] = None)`
:   Resize input data. Aspect ratio is maintained.
    e.g. if image is 1200 x 600 and 300 x 300 is requested the result will be 600 x 300
    
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

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`ScaleBoundingBoxes(name: Optional[str] = None)`
:   Mapping boxes coordinate to scale in original image.
    The coordinate of boxes from detection model is relative to the input image of network, 
    image is scaled and padded/cropped. So we need to do a linear mapping to get the real coordinate of original image.
    input:
        box_of_nms_out: output of NMS, shape [num_boxes, 6]
        original_image: original image decoded from jpg/png<uint8_t>[H, W, 3<BGR>]
        scaled_image: scaled image, but without padding/crop[<uint8_t>[H1, W1, 3<BGR>]
        letter_boxed_image: scaled image and with padding/crop[<uint8_t>[H2, W3, 3<BGR>]
    
    output:
        scaled_box_out: shape [num_boxes, 6] with coordinate mapped to original image.
    
    Args:
        name: Optional name of step. Defaults to 'ScaleBoundingBoxes'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`SelectBestBoundingBoxesByNMS(iou_threshold: float = 0.5, score_threshold: float = 0.67, max_detections: int = 300, name: Optional[str] = None)`
:   Non-maximum suppression (NMS) is to filter out redundant bounding boxes.
    This step is used to warp the boxes and scores into onnx SelectBestBoundingBoxesByNMS op.
    Input:
        boxes:  float[num_boxes, 4]
        scores:  shape float[num_boxes, num_classes]
    
    Output:
        nms_out: float[_few_num_boxes, 6<coordinate+score+class>]
    
    Args:
    Please refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#SelectBestBoundingBoxesByNMS
    for more details about the parameters.
        iou_threshold:  same as SelectBestBoundingBoxesByNMS op, intersection /union of boxes 
        score_threshold:  If this box's score is lower than score_threshold, it will be removed.
        max_detections:  max number of boxes to be selected
        name: Optional name of step. Defaults to 'SelectBestBoundingBoxesByNMS'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`SplitOutBoxAndScore(num_classes: int = 80, name: Optional[str] = None)`
:   Split the output of the model into boxes and scores. This step will also handle the optional object score.
    Input shape: <float>{num_boxes, 4/5+num_classes}
    Output shape: <float>{num_boxes, 4}, <float>{num_boxes, num_classes}
    |x1,x2,x3,x4, (obj), cls_1, ... cls_num|
            /\
           /  \
    |x1,x2,x3,x4|  |cls_1, ... clx_num|*(obj)
    obj is optional, if it is not present, it will be set to 1.0
    This is where 4/5 comes from, '4' represent coordinates and the fifth object probability.
    
    Args:
        num_classes: number of classes
        name: Optional name of step. Defaults to 'SplitOutBoxAndScore'

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