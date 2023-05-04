# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import io
import os
import onnx
import re
import torch
import numpy as np

from onnx import numpy_helper
from transformers import WhisperProcessor

from onnxruntime_extensions import PyOrtFunction, util
from onnxruntime_extensions.cvt import HFTokenizerConverter


# the flags for pre-processing
USE_ONNX_STFT = True
USE_AUDIO_DECODER = True


if not USE_AUDIO_DECODER:
    try:
        import librosa
    except ImportError:
        raise ImportError("Please pip3 install librosa without ort-extensions audio codec support.")


# hard-coded audio hyperparameters
# copied from https://github.com/openai/whisper/blob/main/whisper/audio.py#L12
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH


class CustomOpStftNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(g, self, n_fft, hop_length, window):
        t_n_fft = g.op('Constant', value_t=torch.tensor(n_fft, dtype=torch.int64))
        t_hop_length = g.op('Constant', value_t=torch.tensor(hop_length, dtype=torch.int64))
        t_frame_size = g.op('Constant', value_t=torch.tensor(n_fft, dtype=torch.int64))
        return g.op("ai.onnx.contrib::StftNorm", self, t_n_fft, t_hop_length, window, t_frame_size)

    @staticmethod
    def forward(ctx, audio, n_fft, hop_length, window):
        win_length = window.shape[0]
        stft = torch.stft(audio, n_fft, hop_length, win_length, window,
                          center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        return stft.abs() ** 2


class WhisperPrePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(N_FFT)
        self.mel_filters = torch.from_numpy(util.mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        stft_norm = CustomOpStftNorm.apply(audio_pcm, N_FFT, HOP_LENGTH, self.window)
        magnitudes = stft_norm[:, :, :-1]
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        spec_min = log_spec.max() - 8.0
        log_spec = torch.maximum(log_spec, spec_min)
        spec_shape = log_spec.shape
        padding_spec = torch.ones(spec_shape[0],
                                  spec_shape[1], (N_SAMPLES // HOP_LENGTH - spec_shape[2]), dtype=torch.float)
        padding_spec *= spec_min
        log_spec = torch.cat((log_spec, padding_spec), dim=2)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def _to_onnx_stft(onnx_model):
    """Convert custom-op STFT-Norm to ONNX STFT"""
    node_idx = 0
    new_stft_nodes = []
    stft_norm_node = None
    for node in onnx_model.graph.node:
        if node.op_type == "StftNorm":
            stft_norm_node = node
            break
        node_idx += 1

    if stft_norm_node is None:
        raise RuntimeError("Cannot find STFTNorm node in the graph")

    make_node = onnx.helper.make_node
    replaced_nodes = [
        make_node('Constant', inputs=[], outputs=['const_14_output_0'], name='const_14',
                  value=numpy_helper.from_array(np.array([0, N_FFT // 2, 0, N_FFT // 2], dtype='int64'),
                  name='const_14')),
        make_node('Pad',
                  inputs=[stft_norm_node.input[0], 'const_14_output_0'],
                  outputs=['pad_1_output_0'], mode='reflect'),
        make_node('STFT',
                  inputs=['pad_1_output_0', stft_norm_node.input[2], stft_norm_node.input[3], stft_norm_node.input[4]],
                  outputs=['stft_output_0'], name='stft', domain='', onesided=1),
        make_node('Transpose', inputs=['stft_output_0'], outputs=['transpose_1_output_0'], name='transpose_1',
                  perm=[0, 2, 1, 3]),
        make_node('Constant', inputs=[], outputs=['const_17_output_0'], name='const_17',
                  value=numpy_helper.from_array(np.array([2], dtype='int64'), name='')),
        make_node('Constant', inputs=[], outputs=['const_18_output_0'], name='const_18',
                  value=numpy_helper.from_array(np.array([0], dtype='int64'), name='')),
        make_node('Constant', inputs=[], outputs=['const_19_output_0'], name='const_19',
                  value=numpy_helper.from_array(np.array([-1], dtype='int64'), name='')),
        make_node('Constant', inputs=[], outputs=['const_20_output_0'], name='const_20',
                  value=numpy_helper.from_array(np.array([1], dtype='int64'), name='')),
        make_node('Slice', inputs=['transpose_1_output_0', 'const_18_output_0', 'const_19_output_0',
                                   'const_17_output_0', 'const_20_output_0'], outputs=['slice_1_output_0'],
                  name='slice_1'),
        make_node('Constant', inputs=[], outputs=['const0_output_0'], name='const0', value_int=0),
        make_node('Constant', inputs=[], outputs=['const1_output_0'], name='const1', value_int=1),
        make_node('Gather', inputs=['slice_1_output_0', 'const0_output_0'], outputs=['gather_4_output_0'],
                  name='gather_4', axis=3),
        make_node('Gather', inputs=['slice_1_output_0', 'const1_output_0'], outputs=['gather_5_output_0'],
                  name='gather_5', axis=3),
        make_node('Mul', inputs=['gather_4_output_0', 'gather_4_output_0'], outputs=['mul_output_0'], name='mul0'),
        make_node('Mul', inputs=['gather_5_output_0', 'gather_5_output_0'], outputs=['mul_1_output_0'], name='mul1'),
        make_node('Add', inputs=['mul_output_0', 'mul_1_output_0'], outputs=[stft_norm_node.output[0]], name='add0'),
    ]
    new_stft_nodes.extend(onnx_model.graph.node[:node_idx])
    new_stft_nodes.extend(replaced_nodes)
    new_stft_nodes.extend(onnx_model.graph.node[node_idx + 1:])
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_stft_nodes)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def _torch_export(*arg, **kwargs):
    with io.BytesIO() as f:
        torch.onnx.export(*arg, f, **kwargs)
        return onnx.load_from_string(f.getvalue())


def preprocessing(audio_data):
    if USE_AUDIO_DECODER:
        decoder = PyOrtFunction.from_customop(
            "AudioDecoder", cpu_only=True, downsampling_rate=SAMPLE_RATE, stereo_to_mono=1)
        audio_pcm = torch.from_numpy(decoder(audio_data))
    else:
        audio_pcm = torch.from_numpy(audio_data)

    prep_model_name = 'whisper_pre.onnx'
    whisper_processing = WhisperPrePipeline()

    model_args = (audio_pcm,)
    pre_model = _torch_export(
        whisper_processing,
        model_args,
        input_names=["audio_pcm"],
        output_names=["log_mel"],
        do_constant_folding=True,
        export_params=True,
        opset_version=17,
        dynamic_axes={
            "audio_pcm": {1: "sample_len"},
        }
    )
    onnx.save_model(pre_model, os.path.join(root_dir, prep_model_name))
    if USE_ONNX_STFT:
        pre_model = _to_onnx_stft(pre_model)

    pre_f = PyOrtFunction.from_model(pre_model, cpu_only=True)
    if not USE_AUDIO_DECODER:
        return pre_f(audio_data)
    else:
        pre_full = onnx.compose.merge_models(
            decoder.onnx_model,
            pre_model,
            io_map=[("floatPCM", "audio_pcm")])
        pre_f = PyOrtFunction.from_model(pre_full, cpu_only=True)

        onnx.save_model(pre_f.onnx_model, os.path.join(root_dir, "whisper_codec_pre.onnx"))
        result = pre_f(audio_data)
        return result


def merge_models(core: str, output_model: str, audio_data):
    m_pre_path = os.path.join(root_dir, "whisper_codec_pre.onnx" if USE_AUDIO_DECODER else "whisper_pre.onnx")
    m_pre = onnx.load_model(m_pre_path)
    m_core = onnx.load_model(core)
    m1 = onnx.compose.merge_models(m_pre, m_core, io_map=[("log_mel", "input_features")])
    m2 = onnx.load_model(os.path.join(root_dir, "whisper_post.onnx"))

    m_all = onnx.compose.merge_models(m1, m2, io_map=[("sequences", "ids")])
    bpe_decoder_node = m_all.graph.node.pop(-1)
    make_node = onnx.helper.make_node
    bpe_decoder_node.input.pop(0)
    bpe_decoder_node.input.extend(["generated_ids"])
    m_all.graph.node.extend([
        make_node('Cast', ['sequences'], ["generated_ids"], to=onnx.TensorProto.INT64),
        bpe_decoder_node
        ])
    try:
        onnx.save_model(m_all, output_model)
    except ValueError:
        onnx.save_model(m_all, output_model,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=f"{os.path.basename(output_model)}.data",
                        convert_attribute=True)
    print(f"The final merged model was saved as: {output_model}")

    print("Verify the final model...")
    m_final = PyOrtFunction.from_model(output_model, cpu_only=True)
    output_text = m_final(audio_data,
                          np.asarray([200], dtype=np.int32),
                          np.asarray([0], dtype=np.int32),
                          np.asarray([2], dtype=np.int32),
                          np.asarray([1], dtype=np.int32),
                          np.asarray([1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32),
                          np.zeros((1, N_MELS, N_FRAMES)).astype(np.int32))
    print(output_text)


def postprocessing(token_ids, hf_processor):
    fn_decoder = PyOrtFunction.from_customop(
        "BpeDecoder",
        cvt=HFTokenizerConverter(hf_processor.tokenizer).bpe_decoder,
        skip_special_tokens=True,
        cpu_only=True)

    onnx.save_model(fn_decoder.onnx_model, os.path.join(root_dir, "whisper_post.onnx"))
    return fn_decoder(token_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", required=True, help="Path to audio file")
    parser.add_argument("-m", "--model", required=True, help="Path to custom export of Whisper with beam search")
    args = parser.parse_args()

    print("Looking for the exported model...", end='')
    onnx_model_name = os.path.basename(args.model)
    if not re.search("whisper-.*_beamsearch\.onnx", onnx_model_name):
        print("None")
        print("Cannot find the whisper beamsearch ONNX models. "
              "Please run this script from where Whisper ONNX model was exported. like */onnx_models/openai")
        exit(-1)
    else:
        print(f"{onnx_model_name}")

    model_name = "openai/" + onnx_model_name[:-len("_beamsearch.onnx")]
    root_dir = os.path.dirname(args.model)

    _processor = WhisperProcessor.from_pretrained(model_name)
    # The model similar to Huggingface model like:
    # model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # The onnx model can be generated by the following command:
    #   python <ONNXRUNTIME_DIR>\onnxruntime\python\tools\transformers\models\whisper\convert_to_onnx.py
    #       -m "openai/whisper-base.en" -e
    # !only be valid after onnxruntime 1.15 or main branch of 04/04/2023
    model = PyOrtFunction.from_model(args.model, cpu_only=True)

    test_file = util.get_test_data_file(args.audio)
    if USE_AUDIO_DECODER:
        with open(test_file, "rb") as _f:
            audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
    else:
        audio_blob, _ = librosa.load(test_file)
    audio_blob = np.expand_dims(audio_blob, axis=0)  # add a batch_size dimension

    log_mel = preprocessing(audio_blob)
    print(log_mel.shape)

    input_features = log_mel
    # similar to:
    # generated_ids = model.generate(torch.from_numpy(input_features)).numpy()
    ort_outputs = model(input_features,
                        np.asarray([200], dtype=np.int32),
                        np.asarray([0], dtype=np.int32),
                        np.asarray([2], dtype=np.int32),
                        np.asarray([1], dtype=np.int32),
                        np.asarray([1.0], dtype=np.float32),
                        np.asarray([1.0], dtype=np.float32),
                        np.zeros(input_features.shape).astype(np.int32))
    generated_ids = ort_outputs[0]

    text = postprocessing(generated_ids[0], _processor)
    print(text)

    print("build the final model...")
    merge_models(args.model, args.model.replace("beamsearch", "all"), audio_blob)
