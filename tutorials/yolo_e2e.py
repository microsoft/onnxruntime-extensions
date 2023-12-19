# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy

from pathlib import Path
import onnxruntime_extensions


def get_yolo_model(version: int, onnx_model_name: str):
    # install yolov8
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path(f"yolov{version}n.pt")
    model = ultralytics.YOLO(str(pt_model))  # load a pretrained model
    exported_filename = model.export(format="onnx")  # export the model to ONNX format
    assert exported_filename, f"Failed to export yolov{version}n.pt to onnx"
    import shutil
    shutil.move(exported_filename, onnx_model_name)


def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path):
    """Construct the pipeline for an end2end model with pre and post processing. 
    The final model can take raw image binary as inputs and output the result in raw image file.

    Args:
        input_model_file (Path): The onnx yolo model.
        output_model_file (Path): where to save the final onnx model.
    """
    from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
    add_ppp.yolo_detection(input_model_file, output_model_file, "jpg", onnx_opset=18)


def run_inference(onnx_model_file: Path):
    import onnxruntime as ort
    import numpy as np

    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    image = np.frombuffer(open('../test/data/ppp_vision/wolves.jpg', 'rb').read(), dtype=np.uint8)
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}
    output = session.run(['image_out'], inp)[0]
    output_filename = '../test/data/result.jpg'
    open(output_filename, 'wb').write(output)
    from PIL import Image
    Image.open(output_filename).show()


if __name__ == '__main__':
    # YOLO version. Tested with 5 and 8.
    version = 8
    onnx_model_name = Path(f"../test/data/yolov{version}n.onnx")
    if not onnx_model_name.exists():
        print("Fetching original model...")
        get_yolo_model(version, str(onnx_model_name))

    onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".with_pre_post_processing.onnx")
    print("Adding pre/post processing...")
    add_pre_post_processing_to_yolo(onnx_model_name, onnx_e2e_model_name)
    print("Testing updated model...")
    run_inference(onnx_e2e_model_name)
