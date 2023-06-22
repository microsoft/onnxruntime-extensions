# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import transformers

import numpy as np
from onnxruntime_extensions import OrtPyFunction, gen_processing_models


class TestAutoTokenizer(unittest.TestCase):
    def test_t5_tokenizer(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
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


if __name__ == '__main__':
    unittest.main()
