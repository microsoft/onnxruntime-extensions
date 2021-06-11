import unittest
import numpy as np
from onnxruntime_extensions.eager_op import EagerOp, TextToSentence


def _run_text_to_sentence(input, output, model_path):
    v2str = EagerOp.from_customop(TextToSentence, model = model_path)
    result = v2str(input)
    np.testing.assert_array_equal(result, output)


class TestTextToSentence(unittest.TestCase):

    def test_text_to_case1(self):
        inputs = np.array(["This is the Bling-Fire tokenizer. Autophobia, also called monophobia, isolophobia, or eremophobia, is the specific phobia of isolation. 2007年9月日历表_2007年9月农历阳历一览表-万年历. I saw a girl with a telescope. Я увидел девушку с телескопом."])
        outputs = np.array(["This is the Bling-Fire tokenizer.",
                           "Autophobia, also called monophobia, isolophobia, or eremophobia, is the specific phobia of isolation. 2007年9月日历表_2007年9月农历阳历一览表-万年历.",
                           "I saw a girl with a telescope.",
                           "Я увидел девушку с телескопом."])
        _run_text_to_sentence(input=inputs, output=outputs, model_path="data/default_sentence_break_model.bin")


if __name__ == "__main__":
    unittest.main()
