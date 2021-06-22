from pathlib import Path
import unittest
import numpy as np
from onnxruntime_extensions.eager_op import EagerOp, BlingFireSentenceBreaker

def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))

def _run_blingfire_sentencebreaker(input, output, model_path):
    t2stc = EagerOp.from_customop(BlingFireSentenceBreaker, model=model_path)
    result = t2stc(input)
    np.testing.assert_array_equal(result, output)


class TestBlingFireSentenceBreaker(unittest.TestCase):

    def test_text_to_case1(self):
        inputs = np.array([
                              "This is the Bling-Fire tokenizer. Autophobia, also called monophobia, isolophobia, or eremophobia, is the specific phobia of isolation. 2007年9月日历表_2007年9月农历阳历一览表-万年历. I saw a girl with a telescope. Я увидел девушку с телескопом."])
        outputs = np.array(["This is the Bling-Fire tokenizer.",
                            "Autophobia, also called monophobia, isolophobia, or eremophobia, is the specific phobia of isolation. 2007年9月日历表_2007年9月农历阳历一览表-万年历.",
                            "I saw a girl with a telescope.",
                            "Я увидел девушку с телескопом."])
        _run_blingfire_sentencebreaker(input=inputs, output=outputs, model_path=_get_test_data_file('data', 'default_sentence_break_model.bin'))


if __name__ == "__main__":
    unittest.main()
