import pathlib
import inspect


# some util function for testing and tools

def get_test_data_file(*sub_dirs):
    case_file = inspect.currentframe().f_back.f_code.co_filename
    test_dir = pathlib.Path(case_file).parent
    return str(test_dir.joinpath(*sub_dirs).resolve())


def read_file(path):
    with open(str(path)) as file_content:
        return file_content.read()
