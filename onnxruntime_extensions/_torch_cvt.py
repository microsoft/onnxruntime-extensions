# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
_torch_cvt.py: Data processing graph converted from PyTorch
"""

import io
import onnx
import torch
import numpy as np

from onnx import numpy_helper

from ._ortapi2 import make_onnx_model
from ._cuops import SingleOpGraph
from ._hf_cvt import HFTokenizerConverter
from .util import remove_unused_initializers


class _WhisperHParams:
    SAMPLE_RATE = 16000
    N_FFT = 400
    N_MELS = 80
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = N_SAMPLES // HOP_LENGTH


def _mel_filterbank(
        n_fft: int, n_mels: int = 80, sr=16000, min_mel=0, max_mel=45.245640471924965, dtype=np.float32):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns,
    and it is Slaney normalized mel-scale filterbank.
    """
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=dtype)

    # the centers of the frequency bins for the DFT
    freq_bins = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    mel = np.linspace(min_mel, max_mel, n_mels + 2)
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    log_t = mel >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mel[log_t] - min_log_mel))
    mel_bins = freqs

    mel_spacing = np.diff(mel_bins)

    ramps = mel_bins.reshape(-1, 1) - freq_bins.reshape(1, -1)
    for i in range(n_mels):
        left = -ramps[i] / mel_spacing[i]
        right = ramps[i + 2] / mel_spacing[i + 1]

        # intersect them with each other and zero
        fbank[i] = np.maximum(0, np.minimum(left, right))

    energy_norm = 2.0 / (mel_bins[2: n_mels + 2] - mel_bins[:n_mels])
    fbank *= energy_norm[:, np.newaxis]
    return fbank


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
        self.window = torch.hann_window(_WhisperHParams.N_FFT)
        self.mel_filters = torch.from_numpy(
            _mel_filterbank(
                sr=_WhisperHParams.SAMPLE_RATE,
                n_fft=_WhisperHParams.N_FFT,
                n_mels=_WhisperHParams.N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        stft_norm = CustomOpStftNorm.apply(audio_pcm,
                                           _WhisperHParams.N_FFT,
                                           _WhisperHParams.HOP_LENGTH,
                                           self.window)
        magnitudes = stft_norm[:, :, :-1]
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        spec_min = log_spec.max() - 8.0
        log_spec = torch.maximum(log_spec, spec_min)
        spec_shape = log_spec.shape
        padding_spec = torch.ones(spec_shape[0],
                                  spec_shape[1],
                                  _WhisperHParams.N_SAMPLES // _WhisperHParams.HOP_LENGTH - spec_shape[2],
                                  dtype=torch.float)
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
                  value=numpy_helper.from_array(np.array([0,
                                                          _WhisperHParams.N_FFT // 2, 0,
                                                          _WhisperHParams.N_FFT // 2], dtype='int64'),
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


class WhisperDataProcGraph:
    def __init__(self, processor, **kwargs):
        self.hf_processor = processor
        _opset = kwargs.pop('opset', 17)
        self.opset_version = _opset if _opset else 17

    def pre_processing(self, **kwargs):
        use_audio_decoder = kwargs.pop('USE_AUDIO_DECODER', True)
        use_onnx_stft = kwargs.pop('USE_ONNX_STFT', True)
        whisper_processing = WhisperPrePipeline()

        audio_pcm = torch.rand((1, 32000), dtype=torch.float32)
        model_args = (audio_pcm,)
        pre_model = _torch_export(
            whisper_processing,
            model_args,
            input_names=["audio_pcm"],
            output_names=["log_mel"],
            do_constant_folding=True,
            export_params=True,
            opset_version=self.opset_version,
            dynamic_axes={
                "audio_pcm": {1: "sample_len"},
            }
        )
        if use_onnx_stft:
            pre_model = _to_onnx_stft(pre_model)
            remove_unused_initializers(pre_model.graph)

        pre_full = pre_model
        if use_audio_decoder:
            audecoder_g = SingleOpGraph.build_graph(
                "AudioDecoder", downsampling_rate=_WhisperHParams.SAMPLE_RATE, stereo_to_mono=1)
            audecoder_m = make_onnx_model(audecoder_g)
            pre_full = onnx.compose.merge_models(
                audecoder_m,
                pre_model,
                io_map=[("floatPCM", "audio_pcm")])

        return pre_full

    def post_processing(self, **kwargs):
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        g = SingleOpGraph.build_graph(
            "BpeDecoder",
            cvt=HFTokenizerConverter(self.hf_processor.tokenizer).bpe_decoder,
            skip_special_tokens=skip_special_tokens)

        bpenode = g.node[0]
        bpenode.input[0] = "generated_ids"
        nodes = [onnx.helper.make_node('Cast', ['sequences'], ["generated_ids"], to=onnx.TensorProto.INT64),
                 bpenode]
        del g.node[:]
        g.node.extend(nodes)

        inputs = [onnx.helper.make_tensor_value_info("sequences", onnx.TensorProto.INT32, ['N', 'seq_len', 'ids'])]
        del g.input[:]
        g.input.extend(inputs)
        g.output[0].type.CopyFrom(onnx.helper.make_tensor_type_proto(onnx.TensorProto.STRING, ['N', 'text']))

        return make_onnx_model(g, opset_version=self.opset_version)
