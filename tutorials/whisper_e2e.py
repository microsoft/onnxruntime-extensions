# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import onnx
import numpy
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# from onnx import compose
from pathlib import Path
from onnxruntime_extensions import PyOrtFunction, util
from onnxruntime_extensions.cvt import HFTokenizerConverter


# the flags for pre-processing
USE_ONNX_STFT = False
USE_ONNX_COREMODEL = True
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


class CustomOpStft(torch.autograd.Function):
    @staticmethod
    def symbolic(g, self, n_fft, hop_length, window):
        t_frame_step = g.op('Constant', value_t=torch.tensor(hop_length, dtype=torch.int64))
        t_frame_size = g.op('Constant', value_t=torch.tensor(n_fft, dtype=torch.int64))
        return g.op("STFT", self, t_frame_step, window, t_frame_size)

    @staticmethod
    def forward(ctx, audio, n_fft, hop_length, window):
        win_length = window.shape[0]
        stft = torch.stft(audio, n_fft, hop_length, win_length, window,
                          center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        stft = torch.permute(stft, (0, 2, 1))
        return torch.view_as_real(stft)


class WhisperPrePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(N_FFT)
        self.mel_filters = torch.from_numpy(util.mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        if USE_AUDIO_DECODER:
            audio_pcm = audio_pcm.squeeze(0)

        pad_len = N_SAMPLES - audio_pcm.shape[0]
        audio_pcm = torch.nn.functional.pad(audio_pcm, (0, pad_len), mode='constant', value=0)
        audio_pcm = audio_pcm.unsqueeze(0)

        if USE_ONNX_STFT:
            stft = CustomOpStft.apply(audio_pcm, N_FFT, HOP_LENGTH, self.window)
            stft_norm = stft[..., 0] ** 2 + stft[..., 1] ** 2
            stft_norm = torch.permute(stft_norm, (0, 2, 1))
        else:
            stft_norm = CustomOpStftNorm.apply(audio_pcm, N_FFT, HOP_LENGTH, self.window)

        stft_norm.squeeze_(0)
        magnitudes = stft_norm[:, :-1]
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def preprocessing(audio_data):
    if USE_AUDIO_DECODER:
        decoder = PyOrtFunction.from_customop("AudioDecoder")
        audio_pcm = torch.from_numpy(decoder(audio_data.unsqueeze_(0).numpy()))
    else:
        audio_pcm = torch.from_numpy(audio_data)

    prep_model_name = Path('whisper_pre.onnx')
    WhisperProcessing = WhisperPrePipeline()

    model_args = (audio_pcm,)
    torch.onnx.export(
        WhisperProcessing,
        model_args,
        f=str(prep_model_name),
        input_names=["audio_pcm"],
        output_names=["log_mel"],
        do_constant_folding=True,
        export_params=True,
        opset_version=17,
        dynamic_axes={
            "audio_pcm": {0: "samp_len"},
        }
    )

    pre_f = PyOrtFunction.from_model(str(prep_model_name))
    if not USE_AUDIO_DECODER:
        return pre_f(audio_pcm.numpy())
    else:
        # pre_full = compose.merge_models(decoder.onnx_model, 
        #                                 onnx.load_model("whisper_pre.onnx"),
        #                                 io_map=[("floatPCM", "audio_pcm")])
        # pre_f = PyOrtFunction.from_model(pre_full)

        # onnx.compose has some bugs above, so we use the following workaround
        import copy
        new_graph_node = copy.deepcopy(pre_f.onnx_model.graph.node)
        del pre_f.onnx_model.graph.input[:]
        pre_f.onnx_model.graph.input.extend(decoder.onnx_model.graph.input)
        decoder.onnx_model.graph.node[0].output[0] = "audio_pcm"
        del pre_f.onnx_model.graph.node[:]
        pre_f.onnx_model.graph.node.extend(decoder.onnx_model.graph.node)
        pre_f.onnx_model.graph.node.extend(new_graph_node)
        onnx.save_model(pre_f.onnx_model, "whisper_aud_pre.onnx")
        pre_f = PyOrtFunction.from_model("whisper_aud_pre.onnx")
        return pre_f(audio_data.numpy())


def postprocessing(token_ids, hf_processor):
    fn_decoder = PyOrtFunction.from_customop(
        "BpeDecoder",
        cvt=HFTokenizerConverter(hf_processor.tokenizer).bpe_decoder,
        skip_special_tokens=True)

    onnx.save_model(fn_decoder.onnx_model, "whisper_post.onnx")
    return fn_decoder(token_ids)


if __name__ == '__main__':
    print("checking the model...")
    model_name = "openai/whisper-base.en"
    onnx_model_name = "whisper-base.en_beamsearch.onnx"
    if not Path(onnx_model_name).is_file():
        raise RuntimeError(
            "Please run the script from where Whisper ONNX model was exported. like */onnx_models/openai")
    
    _processor = WhisperProcessor.from_pretrained(model_name)
    if USE_ONNX_COREMODEL:
        # The onnx model can be gereated by the following command:
        #   python <ONNXRUNTIME_DIR>\onnxruntime\python\tools\transformers\models\whisper\convert_to_onnx.py
        #       -m "openai/whisper-base.en" -e
        # !only be valid after onnxruntime 1.15 or main branch of 04/04/2023
        model = PyOrtFunction.from_model(onnx_model_name)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

    test_file = util.get_test_data_file("../test/data", "1272-141231-0002.mp3")
    if USE_AUDIO_DECODER:
        with open(test_file, "rb") as _f:
            audio_data = torch.asarray(list(_f.read()), dtype=torch.uint8)
    else:
        audio_data, _ = librosa.load(test_file)

    log_mel = preprocessing(audio_data)
    print(log_mel.shape)

    input_features = numpy.expand_dims(log_mel, axis=0)
    if USE_ONNX_COREMODEL:
        ort_outputs = model(input_features, numpy.asarray([200]),
                            numpy.asarray([0]), numpy.asarray([2]), numpy.asarray([1]),
                            numpy.asarray([1.0], dtype=numpy.float32), numpy.asarray([1.0], dtype=numpy.float32),
                            numpy.zeros(input_features.shape).astype(numpy.int32))
        generated_ids = ort_outputs[0]
    else:
        generated_ids = model.generate(torch.from_numpy(input_features)).numpy()

    text = postprocessing(generated_ids[0], _processor)
    print(text)
