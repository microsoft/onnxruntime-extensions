# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import onnx.shape_inference
import onnxruntime_extensions
from onnxruntime_extensions.tools.pre_post_processing import *
from pathlib import Path
from PIL import Image, ImageDraw

def get_yolov8_pose_model(onnx_model_name: str):
    # install yolov8
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path("yolov8n-pose.pt")
    model = ultralytics.YOLO(str(pt_model))  # load a pretrained model
    success = model.export(format="onnx")  # export the model to ONNX format
    assert success, "Failed to export yolov8n-pose.pt to onnx"
    import shutil
    shutil.move(pt_model.with_suffix('.onnx'), onnx_model_name)


def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path,
                                    output_image: bool = False):
    """Construct the pipeline for an end2end model with pre and post processing. 
    The final model can take raw image binary as inputs and output the result in raw image file.

    Args:
        input_model_file (Path): The onnx yolo model.
        output_model_file (Path): where to save the final onnx model.
        output_image (bool): Model will draw bounding boxes on the original image and output that. It will NOT draw
            the keypoints as there's no custom operator to handle that currently.
            If false, the output will have the same shape as the original model, with all the co-ordinates updated
            to match the original input image.
    """
    if not Path(input_model_file).is_file():
        print("Fetching the model...")
        get_yolov8_pose_model(str(input_model_file))

    print("Adding pre/post processing to the model...")
    model = onnx.load(str(input_model_file.resolve(strict=True)))
    model_with_shape_info = onnx.shape_inference.infer_shapes(model)

    # TEMPORARY TEST of CHW byte input with no use of ConvertImageToBGR
    # inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]
    inputs = [create_named_value("decoded_image", onnx.TensorProto.UINT8, [3, "in_H", "in_W"])]

    model_input_shape = model_with_shape_info.graph.input[0].type.tensor_type.shape
    model_output_shape = model_with_shape_info.graph.output[0].type.tensor_type.shape

    # infer the input sizes from the model.
    w_in = model_input_shape.dim[-1].dim_value
    h_in = model_input_shape.dim[-2].dim_value
    assert(w_in == 640 and h_in == 640)  # expected values

    # output is [1, 56, 8400]
    # there are
    classes_masks_out = model_output_shape.dim[1].dim_value
    boxes_out = model_output_shape.dim[2].dim_value
    assert(classes_masks_out == 56)
    assert(boxes_out == 8400)

    onnx_opset = 18
    pipeline = PrePostProcessor(inputs, onnx_opset)

    # precess steps are responsible for converting any jpg/png image to CHW BGR float32 tensor
    # jpg-->BGR(Image Tensor)--> Resize (scaled Image)-->LetterBox (Fix sized Image)-->(from HWC to)CHW-->float32-->1CHW

    # layout of image prior to Resize and LetterBox being run. post-processing needs to know this to determine where
    # to get the original H and W from
    # Use "HWC" if first step is ConvertImageToBGR
    decoded_image_layout = "CHW"

    pipeline.add_pre_processing(
        [
            Identity(name="OriginalRGBImage"),
            # ConvertImageToBGR(name="OriginalImage"),  # jpg/png image to BGR in HWC layout
            # Resize an arbitrary sized image to a fixed size in not_larger policy
            Resize((h_in, w_in), policy='not_larger', layout=decoded_image_layout),
            # padding or cropping the image to (h_in, w_in)
            LetterBox(target_shape=(h_in, w_in), layout=decoded_image_layout),
            # ChannelsLastToChannelsFirst(),  # HWC to CHW
            ImageBytesToFloat(),  # Convert to float in range 0..1
            Unsqueeze([0]),  # add batch, CHW --> 1CHW
        ]
    )

    # NMS and drawing boxes
    post_processing_steps = [
        Squeeze([0]),  # - Squeeze to remove batch dimension from [batch, 56, 8200] output
        Transpose([1, 0]),  # reverse so box (4)/score (1)/mask (56) is inner dim
        # split the 56 elements into the box, score for the 1 class, and mask info (17 locations x 3 values)
        Split(num_outputs=3, axis=1, splits=[4, 1, 51]),
        # Apply NMS to select best boxes. iou and score values match
        # https://github.com/ultralytics/ultralytics/blob/e7bd159a44cf7426c0f33ed9b413ef4439505a03/ultralytics/models/yolo/pose/predict.py#L34-L35
        SelectBestBoundingBoxesByNMS(iou_threshold=0.7, score_threshold=0.25, has_mask_data=True),
        # Scale boxes and key point coords back to original image. Mask data has 17 key points per box.
        (ScaleNMSBoundingBoxesAndKeyPoints(num_key_points=17, layout=decoded_image_layout),
         [
             # A default connection from SelectBestBoundingBoxesByNMS for input 0
             # A connection from original image to ScaleBoundingBoxes
             # A connection from the resized image to ScaleBoundingBoxes
             # A connection from the LetterBoxed image to ScaleBoundingBoxes
             # We can use the three images to calculate the scale factor and offset.
             # With scale and offset, we can scale the bounding box and key points back to the original image.
             utils.IoMapEntry("OriginalRGBImage", producer_idx=0, consumer_idx=1),
             utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
             utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
        ]),
    ]

    if output_image:
        # separate out the bounding boxes from the keypoint data to use the existing steps/custom op to draw the
        # bounding boxes.
        post_processing_steps += [
            Split(num_outputs=2, axis=-1, splits=[6, 51], name="SplitScaledBoxesAndKeypoints"),
            (DrawBoundingBoxes(mode='CENTER_XYWH', num_classes=1, colour_by_classes=True),
             [
                 utils.IoMapEntry("OriginalRGBImage", producer_idx=0, consumer_idx=0),
                 utils.IoMapEntry("SplitScaledBoxesAndKeypoints", producer_idx=0, consumer_idx=1),
             ]),
            # Encode to jpg/png
            ConvertBGRToImage(image_format="png"),
        ]

    pipeline.add_post_processing(post_processing_steps)

    new_model = pipeline.run(model)

    print("Pre/post proceessing added.")
    # run shape inferencing to validate the new model. shape inferencing will fail if any of the new node
    # types or shapes are incorrect. infer_shapes returns a copy of the model with ValueInfo populated,
    # but we ignore that and save new_model as it is smaller due to not containing the inferred shape information.
    _ = onnx.shape_inference.infer_shapes(new_model, strict_mode=True)
    onnx.save_model(new_model, str(output_model_file.resolve()))
    print("Updated model saved.")


def run_inference(onnx_model_file: Path, output_image: bool = False):
    import onnxruntime as ort
    import numpy as np

    print("Running the model to validate output.")

    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    input_image_path = './data/bus.jpg'
    image_bytes = np.frombuffer(open(input_image_path, 'rb').read(), dtype=np.uint8)
    input_name = [i.name for i in session.get_inputs()]

    # TEMPORARY: Test CHW input
    # model_input = {input_name[0]: image_bytes}
    rgb_image = np.array(Image.open(input_image_path).convert('RGB'))
    rgb_image = rgb_image.transpose((2, 0, 1))  # Channels first
    model_input = {input_name[0]: rgb_image}
    model_output = ['image_out'] if output_image else ['nms_output_with_scaled_boxes_and_keypoints']
    outputs = session.run(model_output, model_input)

    if output_image:
        image_out = outputs[0]
        from io import BytesIO
        s = BytesIO(image_out)
        Image.open(s).show()
    else:
        # manually draw the bounding boxes and skeleton just to prove it works
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        # open original image so we can draw on it
        input_image = Image.open(input_image_path).convert('RGB')
        input_image_draw = ImageDraw.Draw(input_image)

        scaled_nms_output = outputs[0]
        for result in scaled_nms_output:
            # split the 4 box coords, 1 score, 1 class (ignored), keypoints
            (box, score, _, keypoints) = np.split(result, (4, 5, 6))
            keypoints = keypoints.reshape((17, 3))

            # convert box from centered XYWH to co-ords and draw rectangle
            # NOTE: The pytorch model seems to output XYXY co-ords. Not sure why that's different.
            half_w = (box[2] / 2)
            half_h = (box[3] / 2)
            x0 = box[0] - half_w
            y0 = box[1] - half_h
            x1 = box[0] + half_w
            y1 = box[1] + half_h
            input_image_draw.rectangle(((x0, y0), (x1, y1)), outline='red', width=4)

            # draw skeleton
            # See https://github.com/ultralytics/ultralytics/blob/e7bd159a44cf7426c0f33ed9b413ef4439505a03/ultralytics/utils/plotting.py#L171
            for i, sk in enumerate(skeleton):
                # convert keypoint index in `skeleton` to 0-based index and get keypoint data for it
                keypoint1 = keypoints[sk[0] - 1]
                keypoint2 = keypoints[sk[1] - 1]
                pos1 = (int(keypoint1[0]), int(keypoint1[1]))
                pos2 = (int(keypoint2[0]), int(keypoint2[1]))
                conf1 = keypoint1[2]
                conf2 = keypoint2[2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue

                def coord_valid(coord):
                    x, y = coord
                    return 0 <= x < input_image.width and 0 <= y < input_image.height

                if coord_valid(pos1) and coord_valid(pos2):
                    input_image_draw.line((pos1, pos2), fill='yellow', width=2)

        print("Displaying original image with bounding boxes and skeletons.")
        input_image.show()


if __name__ == '__main__':
    onnx_model_name = Path("./data/yolov8n-pose.onnx")
    onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".with_pre_post_processing.onnx")

    # default output is the scaled non-max suppresion data which matches the original model.
    # each result has bounding box (4), score (1), class (1), keypoints(17 x 3) = 57 elements
    # bounding box is centered XYWH format.
    # alternative is to output the original image with the bounding boxes but no key points drawn.
    output_image_with_bounding_boxes = False
    add_pre_post_processing_to_yolo(onnx_model_name, onnx_e2e_model_name, output_image_with_bounding_boxes)
    run_inference(onnx_e2e_model_name, output_image_with_bounding_boxes)
