# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Run the whisper end-to-end inference with ONNXRuntime-Extensions for pre/post processing.
# THIS SCRIPT IS USED TO DEMO ONLY, WHICH IS NOT A PART OF THE PACKAGE.
# TO GENERATE THE FULL-FUNCTION MODEL, PLEASE USE https://github.com/microsoft/Olive
import os
import onnx
import subprocess
import numpy as np
import onnxruntime as ort

from packaging import version
from transformers import WhisperProcessor
from onnxruntime_extensions import OrtPyFunction, util
from onnxruntime_extensions.cvt import gen_processing_models

# Constants
MODEL_NAME = "openai/whisper-tiny.en"
CACHE_DIR = 'temp_caches_onnx'
OUTPUT_DIR = 'temp_model_onnx'
FINAL_MODEL = "whisper_onnx_tiny_en_fp32_e2e.onnx"
TEST_AUDIO_FILE = util.get_test_data_file('../test/data', "1272-141231-0002.mp3")


def check_onnx_version():
    if version.parse(ort.__version__) < version.parse("1.16.0"):
        raise RuntimeError("ONNXRuntime version must >= 1.16.0")


def export_onnx_model():
    print("Exporting Whisper ONNX model from Huggingface model hub...")
    command = ['python', '-m',
               'onnxruntime.transformers.models.whisper.convert_to_onnx',
               '-m', MODEL_NAME,
               '--cache_dir', CACHE_DIR,
               '--output', OUTPUT_DIR,
               '--precision', 'fp32']
    process = subprocess.run(command)
    if process.returncode != 0:
        raise RuntimeError("Failed to export the core ONNX models.")


def process_test_file():
    if not os.path.exists(TEST_AUDIO_FILE):
        raise FileNotFoundError(f"Test audio path {TEST_AUDIO_FILE} does not exist.")

    raw_audio = np.fromfile(TEST_AUDIO_FILE, dtype=np.uint8)
    _processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    pre_m, post_m = gen_processing_models(_processor,
                                          pre_kwargs={"USE_AUDIO_DECODER": True, "USE_ONNX_STFT": True},
                                          post_kwargs={},
                                          opset=17)

    fn_pre = OrtPyFunction.from_model(pre_m, session_options={"graph_optimization_level": 0})
    return fn_pre(np.expand_dims(raw_audio, axis=0)), pre_m, post_m


def main():
    check_onnx_version()
    export_onnx_model()
    log_mel, pre_m, post_m = process_test_file()

    # Apply core ONNX model
    fn_core = OrtPyFunction.from_model(os.path.join(OUTPUT_DIR, "whisper-tiny.en_beamsearch.onnx"), cpu_only=True)
    token_seq = fn_core(log_mel,
                        np.asarray([200], dtype=np.int32),
                        np.asarray([0], dtype=np.int32),
                        np.asarray([2], dtype=np.int32),
                        np.asarray([1], dtype=np.int32),
                        np.asarray([1.0], dtype=np.float32),
                        np.asarray([1.0], dtype=np.float32))
    print(token_seq.shape)

    # Apply post processing
    fn_post = OrtPyFunction.from_model(post_m, cpu_only=True)
    output_text = fn_post(token_seq)
    print(output_text)

    # Merge models and save final model
    print("Combine the data processing graphs into the ONNX model...")
    final_m = util.quick_merge(pre_m, fn_core.onnx_model, post_m)
    onnx.save(final_m, FINAL_MODEL)

    # Test the final model
    raw_audio = np.fromfile(TEST_AUDIO_FILE, dtype=np.uint8)
    text = OrtPyFunction.from_model(final_m, cpu_only=True)(
        np.expand_dims(raw_audio, axis=0),
        np.asarray([200], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32))
    print(text)


if __name__ == "__main__":
    main()
