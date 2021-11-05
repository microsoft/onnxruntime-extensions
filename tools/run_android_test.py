import contextlib
import os
import utils.android as android
from utils import (run, is_windows, is_macOS, is_linux, get_logger)


def run_subprocess(args, cwd=None, capture_stdout=False, dll_path=None,
                   shell=False, env={}, python_path=None):
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path
    if python_path:
        if is_windows():
            my_env["PYTHONPATH"] = python_path + os.pathsep + my_env["PYTHONPATH"]
        else:
            if "PYTHONPATH" in my_env:
                my_env["PYTHONPATH"] += os.pathsep + python_path
            else:
                my_env["PYTHONPATH"] = python_path

    my_env.update(env)

    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell, env=my_env)


def run_android_tests(args, source_dir, build_dir, config, cwd):
    sdk_tool_paths = android.get_sdk_tool_paths(args.android_sdk_path)
    device_dir = '/data/local/tmp'

    def adb_push(src, dest, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, 'push', src, dest], **kwargs)

    def adb_shell(*args, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, 'shell', *args], **kwargs)

    def adb_install(*args, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, 'install', *args], **kwargs)

    def run_adb_shell(cmd):
        # GCOV_PREFIX_STRIP specifies the depth of the directory hierarchy to strip and
        # GCOV_PREFIX specifies the root directory
        # for creating the runtime code coverage files.
        if args.code_coverage:
            adb_shell(
                'cd {0} && GCOV_PREFIX={0} GCOV_PREFIX_STRIP={1} {2}'.format(
                    device_dir, cwd.count(os.sep) + 1, cmd))
        else:
            adb_shell('cd {} && {}'.format(device_dir, cmd))

    if args.android_abi == 'x86_64':
        with contextlib.ExitStack() as context_stack:
            if args.android_run_emulator:
                avd_name = "ort_android"
                system_image = "system-images;android-{};google_apis;{}".format(
                    args.android_api, args.android_abi)

                android.create_virtual_device(sdk_tool_paths, system_image, avd_name)
                emulator_proc = context_stack.enter_context(
                    android.start_emulator(
                        sdk_tool_paths=sdk_tool_paths,
                        avd_name=avd_name,
                        extra_args=[
                            "-partition-size", "2047",
                            "-wipe-data"]))
                context_stack.callback(android.stop_emulator, emulator_proc)

            adb_push('testdata', device_dir, cwd=cwd)
            adb_push(
                os.path.join(source_dir, 'cmake', 'external', 'onnx', 'onnx', 'backend', 'test'),
                device_dir, cwd=cwd)
            adb_push('onnxruntime_test_all', device_dir, cwd=cwd)
            adb_shell('chmod +x {}/onnxruntime_test_all'.format(device_dir))
            adb_push('onnx_test_runner', device_dir, cwd=cwd)
            adb_shell('chmod +x {}/onnx_test_runner'.format(device_dir))
            run_adb_shell('{0}/onnxruntime_test_all'.format(device_dir))

            if args.build_java:
                gradle_executable = 'gradle'
                # use the gradle wrapper if it exists, the gradlew should be setup under <repo root>/java
                gradlew_path = os.path.join(source_dir, 'java',
                                            'gradlew.bat' if is_windows() else 'gradlew')
                if os.path.exists(gradlew_path):
                    gradle_executable = gradlew_path
                android_test_path = os.path.join(cwd, "java", "androidtest", "android")
                run_subprocess([gradle_executable, '--no-daemon',
                                '-DminSdkVer={}'.format(args.android_api),
                                'clean', 'connectedDebugAndroidTest'],
                               cwd=android_test_path)

            if args.use_nnapi:
                adb_shell('cd {0} && {0}/onnx_test_runner -e nnapi {0}/test'.format(device_dir))
            else:
                adb_shell('cd {0} && {0}/onnx_test_runner {0}/test'.format(device_dir))
            # run shared_lib_test if necessary
            if args.build_shared_lib:
                adb_push('libonnxruntime.so', device_dir, cwd=cwd)
                adb_push('onnxruntime_shared_lib_test', device_dir, cwd=cwd)
                adb_shell('chmod +x {}/onnxruntime_shared_lib_test'.format(device_dir))
                run_adb_shell(
                    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{0} && {0}/onnxruntime_shared_lib_test'.format(
                        device_dir))
