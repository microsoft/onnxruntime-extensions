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
from transformers import WhisperConfig, WhisperProcessor
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


def get_model_inputs(ort_session, audio_data):
    config = WhisperConfig.from_pretrained(MODEL_NAME)
    ort_names = list(map(lambda entry: entry.name, ort_session.get_inputs()))
    print(ort_names)

    inputs = [
        audio_data,                          # audio_stream/input_features
        np.asarray([200], dtype=np.int32),   # max_length
        np.asarray([0], dtype=np.int32),     # min_length
        np.asarray([2], dtype=np.int32),     # num_beams
        np.asarray([1], dtype=np.int32),     # num_return_sequences
        np.asarray([1.0], dtype=np.float32), # length_penalty
        np.asarray([1.0], dtype=np.float32), # repetition_penalty
    ]
    required_input_names = {"audio_stream", "input_features", "max_length", "min_length", "num_beams",
                            "num_return_sequences", "length_penalty", "repetition_penalty"}

    # Add optional inputs if present in model
    batch_size = 1
    N_MELS = 80, N_FRAMES = 3000
    for name in ort_names:
        if name in required_input_names:
            continue
        elif name == "vocab_mask":
            inputs.append(np.ones(config.vocab_size, dtype=np.int32))
        elif name == "prefix_vocab_mask":
            inputs.append(np.ones((batch_size, config.vocab_size), dtype=np.int32))
        elif name == "attention_mask":
            # For older ORT versions that have the dummy attention mask input for the beam search op
            inputs.append(np.zeros((batch_size, N_MELS, N_FRAMES), dtype=np.int32))
        elif name == "decoder_input_ids":
            inputs.append(np.array([[config.decoder_start_token_id]], dtype=np.int32))
        elif name == "logits_processor":
            inputs.append(np.array([1], dtype=np.int32))
        else:
            raise NotImplementedError(f"The '{name}' input name needs to be added")
    return inputs


def main():
    check_onnx_version()
    export_onnx_model()
    log_mel, pre_m, post_m = process_test_file()

    # Apply core ONNX model
    fn_core = OrtPyFunction.from_model(os.path.join(OUTPUT_DIR, "whisper-tiny.en_beamsearch.onnx"), cpu_only=True)
    fn_core_ort_session = fn_core._ensure_ort_session()
    model_inputs = get_model_inputs(fn_core_ort_session, log_mel)
    token_seq = fn_core(*model_inputs)
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
    raw_audio = np.expand_dims(raw_audio, axis=0)
    e2e_model = OrtPyFunction.from_model(final_m, cpu_only=True)
    e2e_model_ort_session = e2e_model._ensure_ort_session()
    model_inputs = get_model_inputs(e2e_model_ort_session, raw_audio)
    text = e2e_model(*model_inputs)
    print(text)


if __name__ == "__main__":
    main()
