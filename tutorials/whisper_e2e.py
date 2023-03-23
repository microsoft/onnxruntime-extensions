import onnx
import numpy
import torch
import librosa
from transformers import WhisperProcessor


from pathlib import Path
from onnxruntime_extensions import PyOrtFunction
from onnxruntime_extensions.cvt import HFTokenizerConverter

# hard-coded audio hyperparameters
# copied from https://github.com/openai/whisper/blob/main/whisper/audio.py#L12
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH


# torch tokenizer and models
_processor = None


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
        self.mel_filters = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))

    def forward(self, audio_pcm: torch.Tensor):
        pad_len = N_SAMPLES - audio_pcm.shape[0]
        audio_pcm = torch.nn.functional.pad(audio_pcm, (0, pad_len), mode='constant', value=0)
        audio_pcm = audio_pcm.unsqueeze(0)

        USE_ONNX_STFT = True

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


def preprocessing(audio_pcm):
    prep_model_name = Path('whisper_pre.onnx')
    WhisperProcessing = WhisperPrePipeline()
    audio_pcm = torch.randn(N_SAMPLES).type(torch.float32)

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
    return pre_f(audio_pcm.numpy())


def postprocessing(token_ids):
    fn_decoder = PyOrtFunction.from_customop(
        "BpeDecoder",
        cvt=HFTokenizerConverter(_processor.tokenizer).bpe_decoder,
        skip_special_tokens=False)

    onnx.save_model(fn_decoder.onnx_model, "whisper_post.onnx")
    return fn_decoder(token_ids)


if __name__ == '__main__':

    print("preparing the model...")
    _processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    # create a fake tensor to create the model.
    audio_pcm = torch.rand(16000, dtype=torch.float32)

    # TODO: Add a audio recording here.

    log_mel = preprocessing(audio_pcm)
    print(log_mel.shape)

    # TODO: temporarily create a fixed output to demo the post-process, will be removed later if
    # onnx model with beam search model is ready.
    tokens = _processor.tokenizer.tokenize("I was born in 92000, and this is falsé.")
    ids = _processor.tokenizer.convert_tokens_to_ids(tokens)
    text = postprocessing(numpy.asarray(ids, dtype=numpy.int64))
    print(text)
