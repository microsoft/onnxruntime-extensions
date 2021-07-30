import os
import unittest
from pathlib import Path
from onnxruntime_extensions import cmake_helper


def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))


class TestCMakeHelper(unittest.TestCase):
    def test_cmake_file_gen(self):
        cfgfile = _get_test_data_file('data', 'test.op.config')
        cfile = 'ocos_enabledoplist.cmake'
        cmake_helper.gen_cmake_oplist(cfgfile, cfile)
        found = False
        with open(cfile, 'r') as f:
            for _ln in f:
                if _ln.strip() == "set(OCOS_ENABLE_GPT2_TOKENIZER ON CACHE INTERNAL \"\")":
                    found = True
                    break

        os.remove(cfile)
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
