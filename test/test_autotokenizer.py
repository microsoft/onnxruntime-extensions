# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import transformers as _hfts

import numpy as np
import onnxruntime as _ort
from packaging import version
from onnxruntime_extensions import OrtPyFunction, util, gen_processing_models


class TestAutoTokenizer(unittest.TestCase):
    def test_t5_tokenizer(self):
        tokenizer = _hfts.AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
        ids = tokenizer.encode("best hotel in bay area.", return_tensors="np")
        print(ids)

        alpha = 0
        nbest_size = 0
        flags = 0

        t5_default_inputs = (
            np.array(
                [nbest_size], dtype=np.int64),
            np.array([alpha], dtype=np.float32),
            np.array([flags & 1], dtype=np.bool_),
            np.array([flags & 2], dtype=np.bool_),
            np.array([flags & 4], dtype=np.bool_))

        ort_tok = OrtPyFunction.from_model(gen_processing_models(tokenizer, pre_kwargs={})[0])
        actual_ids = ort_tok(["best hotel in bay area."], *t5_default_inputs)[0]
        np.testing.assert_array_equal(ids[0][:-1], actual_ids)

    @unittest.skipIf(version.parse(_ort.__version__) < version.parse("1.14.0"), "skip for onnxruntime < 1.14.0")
    def test_whisper(self):
        processor = _hfts.WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        pre_m, post_m = gen_processing_models(processor,
                              pre_kwargs={"USE_AUDIO_DECODER": False, "USE_ONNX_STFT": False},
                              post_kwargs={})

        fn_pre = OrtPyFunction.from_model(pre_m, session_options={"graph_optimization_level": 0})
        t = np.linspace(0, 2*np.pi, 480000).astype(np.float32)
        simaudio = np.expand_dims(np.sin(2 * np.pi * 100 * t), axis=0)
        log_mel = fn_pre(simaudio)

        self.assertEqual(log_mel.shape, (1, 80, 3000))

        fn_post = OrtPyFunction.from_model(post_m)
        rel = fn_post(np.asarray([3, 4, 5], dtype=np.int32))
        self.assertEqual(rel[0], "$%&")

    @unittest.skipIf(version.parse(_ort.__version__) < version.parse("1.14.0"), "skip for onnxruntime < 1.14.0")
    def test_whisper_audio_decoder(self):
        processor = _hfts.WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        pre_m, _ = gen_processing_models(processor,
                              pre_kwargs={"USE_AUDIO_DECODER": True, "USE_ONNX_STFT": True})

        fn_pre = OrtPyFunction.from_model(pre_m, session_options={"graph_optimization_level": 0})
        test_flac_file = util.get_test_data_file('data', '1272-141231-0002.flac')
        raw_audio = np.fromfile(test_flac_file, dtype=np.uint8)
        log_mel = fn_pre(np.expand_dims(raw_audio, axis=0))

        self.assertEqual(log_mel.shape, (1, 80, 3000))


if __name__ == '__main__':
    unittest.main()
