import sys
from pathlib import Path
from onnxruntime_extensions import cmake_helper

if __name__ == '__main__':
    # command: python generate_cmake_ops_list.py <path-to-operators-config-file>
    # will generate the _selectedoplist.cmake file in ${PROJECT_SOURCE_DIR}/cmake/ folder
    print('generate_cmake_ops_list.py arguments: ', sys.argv)

    if len(sys.argv) == 2:
        print('Generating _selectedoplist.cmake file to folder: ${PROJECT_SOURCE_DIR}/cmake/')

        current_dir = Path(__file__).parent
        target_cmake_path = str(current_dir.joinpath('../cmake/_selectedoplist.cmake'))
        print('Target cmake file path: ', target_cmake_path)

        cmake_helper.gen_cmake_oplist(sys.argv[1], target_cmake_path)
    else:
        print('generate_cmake_ops_list.py arguments error!')
