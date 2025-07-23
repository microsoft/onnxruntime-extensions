# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import re
from functools import partial
from pathlib import Path
from typing import Literal, Optional, TypeAlias
from warnings import warn

import onnx.shape_inference
import onnxruntime_extensions
from onnxruntime_extensions.tools.pre_post_processing import *
from PIL import Image, ImageDraw

ModelSize: TypeAlias = Optional[Literal["s", "m", "l", "x"]]

BASE_PATH = Path(__file__).parent


def _get_yolov8_world_model(onnx_model_path: Path, classes: list[str], size: ModelSize = None):
    # install yolov8
    from pip._internal import main as pipmain

    try:
        import ultralytics
    except ImportError:
        pipmain(["install", "ultralytics"])
        import ultralytics

    if size is None:
        regex = r"yolov8(.{1})"
        matches = re.search(regex, onnx_model_path.name)
        if matches:
            size = matches.group(1)
        else:
            size = "m"
            warn(f"YOLO size not set and not able to determine yolo world size, defaulting to {size}.")

    # NOTE: only v2 models are exportable
    pt_model = Path(f"yolov8{size}-worldv2.pt")
    # load yolo world pretrained model
    model = ultralytics.YOLOWorld(str(pt_model))
    # set to the right classes
    model.set_classes(classes)
    # export the vocabulary to be used as YOLO model
    pt_model_path = onnx_model_path.with_suffix(".pt")
    model.save(pt_model_path)
    model = ultralytics.YOLO(pt_model_path)
    # export the model to ONNX format
    success = model.export(format="onnx", optimize=True, simplify=True)
    assert success, f"Failed to export {pt_model_path.name} to onnx"
    assert onnx_model_path.exists(), "Falied to export "


def _get_model_and_info(
    input_model_path: Path, classes: list[str], model_size: ModelSize = None
) -> tuple[onnx.ModelProto, int, int]:
    if not input_model_path.is_file():
        print(f"Fetching the model... {str(input_model_path)}")
        _get_yolov8_world_model(input_model_path, classes, size=model_size)

    print("Adding pre/post processing to the model...")
    model = onnx.load(str(input_model_path.resolve(strict=True)))

    model_with_shape_info = onnx.shape_inference.infer_shapes(model)

    model_input_shape = model_with_shape_info.graph.input[0].type.tensor_type.shape
    model_output_shape = model_with_shape_info.graph.output[0].type.tensor_type.shape

    num_classes = len(classes)
    # infer the input sizes from the model.
    w_in = model_input_shape.dim[-1].dim_value
    h_in = model_input_shape.dim[-2].dim_value
    assert w_in == 640 and h_in == 640  # expected values

    # output should be [1, num_classes+4(bbox coords), 8400].
    classes_bbox = model_output_shape.dim[1].dim_value
    boxes_out = model_output_shape.dim[2].dim_value
    assert classes_bbox == 4 + num_classes
    assert boxes_out == 8400

    return model, w_in, h_in


def _update_model(model: onnx.ModelProto, output_model_path: Path, pipeline: PrePostProcessor):
    """
    Update the model by running the pre/post processing pipeline
    @param model: ONNX model to update
    @param output_model_path: Filename to write the updated model to.
    @param pipeline: Pre/Post processing pipeline to run.
    """

    new_model = pipeline.run(model)
    print("Pre/post proceessing added.")

    # run shape inferencing to validate the new model. shape inferencing will fail if any of the new node
    # types or shapes are incorrect. infer_shapes returns a copy of the model with ValueInfo populated,
    # but we ignore that and save new_model as it is smaller due to not containing the inferred shape information.
    _ = onnx.shape_inference.infer_shapes(new_model, strict_mode=True)
    onnx.save_model(new_model, str(output_model_path.resolve()))
    print("Updated model saved.")


def _add_pre_post_processing(
    classes: list[str],
    input_type: Literal["rgb", "image"],
    input_model_path: Path,
    output_model_path: Path,
    output_image_format: Optional[Literal["jpg", "png"]] = None,
    input_shape: Optional[List[Union[int, str]]] = None,
    model_size: ModelSize = None,
):
    """
    Add pre and post processing with model input of jpg or png image bytes or just RGB data.
    Pre-processing will convert the input to the correct height, width and data type for the model.
    Post-processing will select the best bounding boxes using NonMaxSuppression, and scale the selected bounding
    boxes to the original image size.

    The post-processing can alternatively return the original image with the bounding boxes drawn on it
    instead of the scaled bounding box and key point data.

    @param classes: Classes that will be sent as prompt to yolo world model.
    @param input_type: Is the input an image or raw rgb.
    @param input_model_path: Path to ONNX model.
    @param output_model_path: Path to write updated model to.
    @param output_image_format: Optional. Specify 'jpg' or 'png' for the post-processing to return image bytes in that
                                format with the bounding boxes drawn on it.
                                Otherwise the model will return the scaled bounding boxes.
                                Can only be used if input_type is 'image'.
    @param input_shape: Optional. Input shape of RGB data. Must be 3D.
                        First or last value must be 3 (channels first or last).
                        This is required if input_type is 'raw'.
    @param model_size: Size of the yolo model. Valid values ["s", "m", "l", "x"].
                        If None, we automatically detect based on filename.
    """
    num_classes = len(classes)
    model, w_in, h_in = _get_model_and_info(input_model_path, classes, model_size)

    pre_processing_steps = []
    if input_type == "rgb":
        if output_image_format is not None:
            raise ValueError("Model cannot output to image when input_type is 'raw'.")
        if input_shape is None:
            raise ValueError("For input_type 'raw', provide input_shape.")
        elif input_shape[0] == 3:
            layout = "CHW"
        elif input_shape[2] == 3:
            layout = "HWC"
        else:
            raise ValueError("Invalid input shape. Either first or last dimension must be 3.")

        inputs = [create_named_value("rgb_data", onnx.TensorProto.UINT8, input_shape)]
        if layout == "CHW":
            # use Identity so we have an output named RGBImageCHW
            # for ScaleNMSBoundingBoxesAndKeyPoints in the post-processing steps
            pre_processing_steps += [Identity(name="RGBImageCHW")]
        else:
            pre_processing_steps += [ChannelsLastToChannelsFirst(name="RGBImageCHW")]  # HWC to CHW
    else:
        inputs = [create_named_value("image_bytes", onnx.TensorProto.UINT8, ["num_bytes"])]
        pre_processing_steps += [
            ConvertImageToBGR(name="BGRImageHWC"),  # jpg/png image to BGR in HWC layout
            ChannelsLastToChannelsFirst(name="BGRImageCHW"),  # HWC to CHW
        ]

    onnx_opset = 18
    pipeline = PrePostProcessor(inputs, onnx_opset)

    pre_processing_steps += [
        # Resize to match model input. Uses not_larger as we use LetterBox to pad as needed.
        Resize((h_in, w_in), policy="not_larger", layout="CHW"),
        LetterBox(target_shape=(h_in, w_in), layout="CHW"),  # padding or cropping the image to (h_in, w_in)
        ImageBytesToFloat(),  # Convert to float in range 0..1
        Unsqueeze([0]),  # add batch, CHW --> 1CHW
    ]

    pipeline.add_pre_processing(pre_processing_steps)

    # NonMaxSuppression and drawing boxes
    post_processing_steps = [
        Squeeze([0]),  # Squeeze to remove batch dimension from [batch, num_classes+4, 8400] output
        Transpose([1, 0]),  # reverse so result info is inner dim
        # split the 56 elements into the box, score for the 1 class, and mask info (17 locations x 3 values)
        Split(num_outputs=2, axis=1, splits=[4, num_classes]),
        # Apply NMS to select best boxes. iou and score values match
        # https://github.com/ultralytics/ultralytics/blob/e7bd159a44cf7426c0f33ed9b413ef4439505a03/ultralytics/models/yolo/pose/predict.py#L34-L35
        # thresholds are arbitrarily chosen. adjust as needed
        SelectBestBoundingBoxesByNMS(iou_threshold=0.7, score_threshold=0.25),
        # Scale boxes and key point coords back to original image. Mask data has 17 key points per box.
        (
            ScaleNMSBoundingBoxesAndKeyPoints(num_key_points=17, layout="CHW"),
            [
                # A default connection from SelectBestBoundingBoxesByNMS for input 0
                # A connection from original image to input 1
                # A connection from the resized image to input 2
                # A connection from the LetterBoxed image to input 3
                # We use the three images to calculate the scale factor and offset.
                # With scale and offset, we can scale the bounding box and key points back to the original image.
                utils.IoMapEntry(
                    "RGBImageCHW" if input_type == "rgb" else "BGRImageCHW", producer_idx=0, consumer_idx=1
                ),
                utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
                utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
            ],
        ),
    ]

    if output_image_format:
        post_processing_steps += [
            # DrawBoundingBoxes on the original image
            # Model imported from pytorch has CENTER_XYWH format
            # two mode for how to color box,
            #   1. colour_by_classes=True, (colour_by_classes), 2. colour_by_classes=False,(colour_by_confidence)
            (
                DrawBoundingBoxes(mode="CENTER_XYWH", num_classes=num_classes, colour_by_classes=True),
                [
                    utils.IoMapEntry("ConvertImageToBGR", producer_idx=0, consumer_idx=0),
                    utils.IoMapEntry("ScaleBoundingBoxes", producer_idx=0, consumer_idx=1),
                ],
            ),
            # Encode to jpg/png
            ConvertBGRToImage(image_format=output_image_format),
        ]

    pipeline.add_post_processing(post_processing_steps)

    print("Updating model ...")

    _update_model(model, output_model_path, pipeline)


def _run_inference(
    onnx_model_path: Path,
    model_input: str,
    model_outputs_image: bool,
    test_image: Path,
    classes: list[str],
    rgb_layout: Optional[str],
):
    import numpy as np
    import onnxruntime as ort

    print(f"Running the model to validate output using {str(test_image)}.")

    providers = ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = ort.InferenceSession(str(onnx_model_path), providers=providers, sess_options=session_options)

    input_name = [i.name for i in session.get_inputs()]
    if model_input == "image":
        image_bytes = np.frombuffer(open(test_image, "rb").read(), dtype=np.uint8)
        model_input = {input_name[0]: image_bytes}
    else:
        rgb_image = np.array(Image.open(test_image).convert("RGB"))
        if rgb_layout == "CHW":
            rgb_image = rgb_image.transpose((2, 0, 1))  # Channels first
        model_input = {input_name[0]: rgb_image}

    model_output = ["image"] if model_outputs_image else ["nms_output_with_scaled_boxes_and_keypoints"]
    outputs = session.run(model_output, model_input)

    if model_outputs_image:
        # jpg or png with bounding boxes draw
        image_out = outputs[0]
        from io import BytesIO

        s = BytesIO(image_out)
        Image.open(s).show()
    else:
        # open original image so we can draw on it
        input_image = Image.open(test_image).convert("RGB")
        input_image_draw = ImageDraw.Draw(input_image)

        scaled_nms_output = outputs[0]
        for result in scaled_nms_output:
            # split the 4 box coords, 1 score, 1 class
            (box, score, class_id) = np.split(result, (4, 5))
            class_id = int(class_id)
            score = float(score * 100)

            # convert box from centered XYWH to co-ords and draw rectangle
            # NOTE: The pytorch model seems to output XYXY co-ords. Not sure why that's different.
            half_w = box[2] / 2
            half_h = box[3] / 2
            x0 = box[0] - half_w
            y0 = box[1] - half_h
            x1 = box[0] + half_w
            y1 = box[1] + half_h
            input_image_draw.rectangle(((x0, y0), (x1, y1)), outline="red", width=4)
            input_image_draw.text((x0, y0), f"{classes[class_id]}-{score:.2f}%")

        print("Displaying original image with bounding boxes.")
        input_image.show()


def load_classes(args) -> list[str]:
    if args.classes:
        return args.classes
    elif args.classes_file:
        classes_file = Path(args.classes_file)
        with classes_file.open() as fp:
            if classes_file.suffix == ".json":
                import json

                classes = json.load(fp)
            elif classes_file.suffix == ".txt":
                classes = fp.read().splitlines()
            else:
                raise ValueError(f"Invalid file type {classes_file}")
        return classes
    else:
        # default value for data/stormtroopers.jpg
        return ["person", "helmet"]


# python tutorials/yolov8_world_e2e.py yolov8m-worldv2.onnx --infer --input=rgb --input_shape H,W,3
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Add pre and post processing to the YOLOv8 World model. The model can be updated to take either 
        jpg/png bytes as input (--input image), or RGB data (--input rgb).
        NOTE: Use only YOLO WOrld v2 model as that is the only one with export capability.
        By default the post processing will scale the bounding boxes and key points to the original image.
    """
    )
    parser.add_argument("model", type=Path, help="The ONNX YOLOv8 World model.")
    classes_group = parser.add_mutually_exclusive_group()
    classes_group.add_argument(
        "--classes",
        type=lambda x: re.split(r"\s*,\s*", x.strip()),
        default=["person", "helmet"],
        help="List of class names that will be passed to yolo world prompt."
        "Default values 'person,helmet' for data/stormtroopers.jpg",
    )
    classes_group.add_argument(
        "--classes-file", type=Path, help="JSON file containing list of classes that will be set as yolo world prompt."
    )
    parser.add_argument(
        "--size",
        choices=("s", "m", "l", "x"),
        default=None,
        help="Size of yolo world model.",
    )
    parser.add_argument(
        "--updated_onnx_model_path",
        type=Path,
        required=False,
        help="Filename to save the updated ONNX model to. If not provided default to the filename "
        "from --onnx_model_path with '.with_pre_post_processing' before the '.onnx' "
        "e.g. yolov8m-worldv2.onnx -> yolov8m-worldv2.with_pre_post_processing.onnx",
    )
    parser.add_argument(
        "--input",
        choices=("image", "rgb"),
        default="image",
        help="Desired model input format. Image bytes from jpg/png or RGB data.",
    )
    parser.add_argument(
        "--input_shape",
        type=lambda x: [int(dim) if dim.isnumeric() else dim for dim in x.split(",")],
        default=["H", "W", 3],
        required=False,
        help="Shape of RGB input if input is 'rgb'. Provide a comma separated list of 3 dimensions. "
        "Symbolic dimensions are allowed. Either the first or last dimension must be 3 to infer "
        "if layout is HWC or CHW. "
        "examples: channels first with symbolic dims for height and width: --input_shape 3,H,W "
        "or channels last with fixed input shape: --input_shape 384,512,3",
    )
    parser.add_argument(
        "--output_as_image",
        choices=("jpg", "png"),
        required=False,
        help="OPTIONAL. If the input is an image, instead of outputting the scaled bounding boxes "
        "the model will draw the bounding boxes on the original image, convert to the "
        "specified format, and output the updated image bytes.",
    )
    parser.add_argument("--infer", action="store_true", help="Run inference on the model to validate output.")
    parser.add_argument(
        "--test_image",
        type=Path,
        default=BASE_PATH / "data/stormtroopers.jpg",
        help="JPG or PNG image to run model with.",
    )

    args = parser.parse_args()

    classes = load_classes(args)

    num_classes = len(classes)
    assert num_classes > 0, "Requires prompt for YOLO world model."

    if args.output_as_image and args.input == "rgb":
        raise argparse.ArgumentError(
            args.output_as_image, "output_as_image argument can only be used if input is 'image'"
        )

    if args.input_shape and len(args.input_shape) != 3:
        raise argparse.ArgumentError(args.input_shape, "Shape of RGB input must have 3 dimensions.")

    updated_model_path = (
        args.updated_onnx_model_path
        if args.updated_onnx_model_path
        else args.model.with_suffix(suffix=".with_pre_post_processing.onnx")
    )

    # default output is the scaled non-max suppression data which matches the original model.
    # each result has bounding box (4), score (1), class (num_classes), = num_classes+5 elements
    # bounding box is centered XYWH format.
    # alternative is to output the original image with the bounding boxes.
    add_pre_post_processing = partial(
        _add_pre_post_processing,
        classes=classes,
        input_type=args.input,
        input_model_path=args.model,
        output_model_path=updated_model_path,
        model_size=args.size,
    )

    if args.input == "rgb":
        print("Updating model with RGB data as input.")
        add_pre_post_processing(input_shape=args.input_shape)
        rgb_layout = "CHW" if args.input_shape[0] == 3 else "HWC"
    elif args.input == "image":
        print("Updating model with jpg/png image bytes as input.")
        add_pre_post_processing(output_image_format=args.output_as_image)
        rgb_layout = None

    if args.infer:
        _run_inference(
            updated_model_path, args.input, args.output_as_image is not None, args.test_image, classes, rgb_layout
        )
