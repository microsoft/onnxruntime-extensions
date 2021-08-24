import os
import subprocess
import sys
import unittest


class TestCMakeHelper(unittest.TestCase):
    def test_cmake_file_gen(self):
        previous_dir = os.getcwd()

        current_dir = os.path.dirname(__file__)
        operator_config_file = os.path.abspath(os.path.join(current_dir, './data/test.op.config'))
        generated_cmake_file = os.path.abspath(os.path.join(current_dir, '../cmake/_selectedoplist.cmake'))

        cmake_tool_dir = os.path.abspath(os.path.join(current_dir, '../ci_build/tools/'))
        os.chdir(cmake_tool_dir)

        # run cmake_helper.py
        subprocess.run([sys.executable, 'cmake_helper.py', operator_config_file], cwd=cmake_tool_dir)

        found = False
        with open(generated_cmake_file, 'r') as f:
            for _ln in f:
                if _ln.strip() == "set(OCOS_ENABLE_GPT2_TOKENIZER ON CACHE INTERNAL \"\")":
                    found = True
                    break

        os.remove(generated_cmake_file)
        self.assertTrue(found)
        os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
