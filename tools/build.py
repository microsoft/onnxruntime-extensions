#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import sys

from pathlib import Path, PurePosixPath

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "utils"))

from utils import (
    android,
    get_logger,
    is_macOS,
    is_windows,
    run,
)  # noqa: E402

log = get_logger("build")


class UsageError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def _check_python_version():
    required_minor_version = 8
    if (sys.version_info.major, sys.version_info.minor) < (3, required_minor_version):
        raise UsageError(
            f"Invalid Python version. At least Python 3.{required_minor_version} is required. "
            f"Actual Python version: {sys.version}"
        )


_check_python_version()


def _parse_arguments():
    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior - allow multiple arguments per line and handle quotes
        def convert_arg_line_to_args(self, arg_line):
            return shlex.split(arg_line)

    parser = Parser(
        description="ONNXRuntime Extensions Shared Library build driver.",
        usage="""
        There are 3 phases which can be individually selected.

        The Update (--update) phase will run CMake to generate makefiles.
        The Build (--build) phase will build all projects.
        The Test (--test) phase will run all unit tests.

        Default behavior is --update --build --test for native architecture builds.
        Default behavior is --update --build for cross-compiled builds.

        If phases are explicitly specified only those phases will be run.
          e.g. run with `--build` to rebuild without running the update or test phases
        """,

        # files containing arguments can be specified on the command line with "@<filename>" and the arguments within
        # will be included at that point
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    def path_from_env_var(env_var: str):
        env_var_value = os.environ.get(env_var)
        return Path(env_var_value) if env_var_value is not None else None

    # Main arguments
    parser.add_argument("--build_dir", type=Path,
                        # We set the default programmatically as it needs to take into account whether we're
                        # cross-compiling
                        help="Path to the build directory. Defaults to 'build/<target platform>'")
    parser.add_argument("--config", nargs="+", default=["Debug"],
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")

    # Build phases
    parser.add_argument("--update", action="store_true", help="Update makefiles.")

    parser.add_argument("--build", action="store_true", help="Build.")

    parser.add_argument("--test", action="store_true", help="Run tests.")
    parser.add_argument("--skip_tests", action="store_true", help="Skip all tests. Overrides --test.")

    parser.add_argument("--clean", action="store_true",
                        help="Run 'cmake --build --target clean' for the selected config/s.")
    # Build phases end

    parser.add_argument("--parallel", nargs="?", const="0", default="1", type=int,
                        help="Use parallel build. The optional value specifies the maximum number of parallel jobs. "
                             "If the optional value is 0 or unspecified, it is interpreted as the number of CPUs.")

    parser.add_argument("--cmake_extra_defines", nargs="+", action="extend", default=[],
                        help="Add extra definitions to pass to CMake during build system generation. "
                             "These are essentially CMake -D options without the leading -D. "
                             "Multiple name=value defines can be specified, with each separated by a space. "
                             "Quote the name and value if the value contains spaces. "
                             "This option can also be specified multiple times. "
                             "  e.g. --cmake_extra_defines \"Name1=the value\" Name2=value2")
    parser.add_argument("--one_cmake_extra_define", action="append", dest="cmake_extra_defines",
                        help="Add one CMake extra definition, see --cmake_extra_defines. "
                             "This option can be specified multiple times.")

    # Test options
    parser.add_argument("--enable_cxx_tests", action="store_true", help="Enable the C++ unit tests.")
    parser.add_argument("--cxx_code_coverage", action="store_true",
                        help="Run C++ unit tests using vstest.exe to produce code coverage output. Windows only.")

    parser.add_argument("--onnxruntime_version", type=str,
                        help="ONNX Runtime version to fetch for headers and library. Default is 1.10.0.")
    parser.add_argument("--onnxruntime_lib_dir", type=Path,
                        help="Path to directory containing the pre-built ONNX Runtime library if you do not want to "
                             "use the library from the ONNX Runtime release package that is fetched by default.")
    # Build for ARM
    parser.add_argument("--arm", action="store_true",
                        help="[cross-compiling] Create ARM makefiles. Requires --update and no existing cache "
                             "CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument("--arm64", action="store_true",
                        help="[cross-compiling] Create ARM64 makefiles. Requires --update and no existing cache "
                             "CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument("--arm64ec", action="store_true",
                        help="[cross-compiling] Create ARM64EC makefiles. Requires --update and no existing cache "
                             "CMake setup. Delete CMakeCache.txt if needed")

    # Android options
    parser.add_argument("--android", action="store_true", help="Build for Android")
    parser.add_argument("--android_abi", default="arm64-v8a", choices=["armeabi-v7a", "arm64-v8a", "x86", "x86_64"],
                        help="Specify the target Android Application Binary Interface (ABI)")
    parser.add_argument("--android_api", type=int, default=27, help="Android API Level, e.g. 21")
    parser.add_argument("--android_home", type=Path, default=path_from_env_var("ANDROID_HOME"),
                        help="Path to the Android SDK.")
    parser.add_argument("--android_ndk_path", type=Path, default=path_from_env_var("ANDROID_NDK_HOME"),
                        help="Path to the Android NDK. Typically `<Android SDK>/ndk/<ndk_version>`.")
    parser.add_argument("--android_adb_device_serial",
                        help="Device serial argument passed to 'adb -s'. Can be used to select a specific device if "
                             "there is more than one.")

    # macOS/iOS options
    parser.add_argument("--build_apple_framework", action="store_true",
                        help="Build a macOS/iOS framework for ONNX Runtime Extensions.")
    parser.add_argument("--ios", action="store_true", help="build for iOS")
    parser.add_argument("--ios_sysroot", default="",
                        help="Specify the name of the platform SDK to be used. e.g. iphoneos, iphonesimulator")
    parser.add_argument("--ios_toolchain_file", default=f"{REPO_DIR}/cmake/ortext_ios.toolchain.cmake", type=Path,
                        help="Path to ios toolchain file. Default is <repo>/cmake/ortext_ios.toolchain.cmake")
    parser.add_argument("--xcode_code_signing_team_id", default="",
                        help="The development team ID used for code signing in Xcode")
    parser.add_argument("--xcode_code_signing_identity", default="",
                        help="The development identity used for code signing in Xcode")
    parser.add_argument("--apple_arch", default="arm64" if platform.machine() == "arm64" else "x86_64",
                        choices=["arm64", "arm64e", "x86_64"],
                        help="Specify the Target specific architectures for macOS and iOS. "
                             "This is only supported on macOS")
    parser.add_argument("--apple_deploy_target", type=str,
                        help="Specify the minimum version of the target platform (e.g. macOS or iOS). "
                             "This is only supported on macOS")

    # WebAssembly options
    parser.add_argument("--wasm", action="store_true", help="Build for WebAssembly")
    parser.add_argument("--emsdk_path", type=Path,
                        help="Specify path to emscripten SDK. Setup manually with: "
                             "  git clone https://github.com/emscripten-core/emsdk")
    parser.add_argument("--emsdk_version", default="3.1.26", help="Specify version of emsdk")

    # x86 args
    parser.add_argument("--x86", action="store_true",
                        help="[cross-compiling] Create Windows x86 makefiles. Requires --update and no existing cache "
                             "CMake setup. Delete CMakeCache.txt if needed")
    # x64 args
    parser.add_argument("--x64", action="store_true",
                        help="doing nothing, just for compatibility")

    # Arguments needed by CI
    parser.add_argument("--cmake_path", default="cmake", type=Path, help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", type=Path, help="Path to the CTest program.")

    parser.add_argument("--cmake_generator",
                        choices=["Visual Studio 16 2019", "Visual Studio 17 2022", "Ninja", "Unix Makefiles", "Xcode"],
                        default="Visual Studio 17 2022" if is_windows() else "Unix Makefiles",
                        help="Specify the generator that CMake invokes.")

    # Binary size reduction options
    parser.add_argument("--include_ops_by_config", type=Path,
                        help="Only include ops specified in the build that are listed in this config file. "
                             "Format of config file is `domain;opset;op1,op2,... "
                             "  e.g. com.microsoft.extensions;1;ImageDecode,ImageEncode")

    parser.add_argument("--disable_exceptions", action="store_true",
                        help="Disable exceptions to reduce binary size.")

    # Language bindings
    parser.add_argument("--build_java", action="store_true", help="Build Java bindings.")

    args = parser.parse_args()

    # validate Android args
    if args.android:
        if not args.android_ndk_path:
            raise UsageError("--android_ndk_path is required to build for Android")
        if not args.android_home:
            raise UsageError("--android_home is required to build for Android")

        args.android_home = args.android_home.resolve(strict=True)
        args.android_ndk_path = args.android_ndk_path.resolve(strict=True)

        if not args.android_home.is_dir() or not args.android_ndk_path.is_dir():
            raise UsageError("Android home and NDK paths must be directories.")

    # set build directory if not provided
    if not args.build_dir:
        target_sys = platform.system()

        # override if we're cross-compiling
        if args.android:
            target_sys = "Android"
        elif args.ios:
            target_sys = "iOS"
        elif args.arm:
            target_sys = "arm"
        elif args.arm64:
            target_sys = "arm64"
        elif args.arm64ec:
            target_sys = "arm64ec"
        elif platform.system() == "Darwin":
            # also tweak name for mac builds
            target_sys = "macOS"
        elif args.wasm:
            target_sys = "wasm"

        args.build_dir = Path("build/" + target_sys)

    return args


def _is_reduced_ops_build(args):
    return args.include_ops_by_config is not None


def _resolve_executable_path(command_or_path: Path, resolution_failure_allowed: bool = False):
    """
    Returns the absolute path of an executable.
    If `resolution_failure_allowed` is True, returns None if the executable path cannot be found.
    """
    executable_path = shutil.which(str(command_or_path))
    if executable_path is None:
        if resolution_failure_allowed:
            return None
        else:
            raise ValueError(f"Failed to resolve executable path for '{command_or_path}'.")

    return Path(executable_path)


def _get_build_config_dir(build_dir: Path, config: str):
    # build directory per configuration
    return build_dir / config


def _run_subprocess(args: list[str],
                    cwd: Path | None = None,
                    capture_stdout: bool = False,
                    shell: bool = False,
                    env: dict[str, str] | None = None,
                    python_path: Path | None = None):

    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    if env is None:
        env = {}

    my_env = os.environ.copy()

    if python_path:
        python_path = str(python_path.resolve())
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)

    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell, env=my_env)


def _is_cross_compiling_on_apple(args):
    if is_macOS():
        return args.ios or args.apple_arch != platform.machine()

    return False


def _validate_cxx_test_args(args):
    ort_lib_dir = None
    if args.onnxruntime_lib_dir:
        ort_lib_dir = args.onnxruntime_lib_dir.resolve(strict=True)
        if not ort_lib_dir.is_dir():
            raise UsageError("onnxruntime_lib_dir must be a directory")

    return ort_lib_dir


def _generate_selected_ops_config(config_file: Path):
    config_file.resolve(strict=True)
    script = REPO_DIR / "tools" / "gen_selectedops.py"
    _run_subprocess([sys.executable, str(script), str(config_file)])


def _setup_emscripten(args):
    if not args.emsdk_path:
        raise UsageError("emsdk_path must be specified for wasm build")

    emsdk_file = str((args.emsdk_path / ("emsdk.bat" if is_windows() else "emsdk")).resolve(strict=True))

    log.info("Installing emsdk...")
    _run_subprocess([emsdk_file, "install", args.emsdk_version], cwd=args.emsdk_path)
    log.info("Activating emsdk...")
    _run_subprocess([emsdk_file, "activate", args.emsdk_version], cwd=args.emsdk_path)


# run the prebuild to create the curl and openssl libraries for the Azure custom ops
def _android_prebuild(android_abi: str, android_ndk_path: Path, android_api_level: int):
    if is_windows():
        log.info("Skipping Android prebuild on Windows because it is not supported yet.")
        return

    prebuild_dir = REPO_DIR / "prebuild"

    # skip if curl binary is found. that is created as the last stage of the prebuild, so if it was successfully
    # installed in the output dir we assume the prebuild completed previously.
    curl_bin = prebuild_dir / "openssl_for_ios_and_android" / "output" / "android" / ("curl-" + android_abi) / "bin" / "curl"
    if curl_bin.exists():
        log.info(f"Found {curl_bin}. Assuming prebuild has completed previously. Skipping prebuild.")
        return

    # set some environment variables required by the script
    env = {
        'ANDROID_API_LEVEL': str(android_api_level),
        'ANDROID_NDK_ROOT': str(android_ndk_path),
    }

    # adjust some strings to the values expected in the script
    if android_abi == "armeabi-v7a":
        curl_abi = "arm"
    elif android_abi == "arm64-v8a":
        curl_abi = "arm64"
    else:
        curl_abi = android_abi

    prebuild_cmd = [
        "/bin/bash",
        "build_curl_for_android.sh",
        curl_abi,
    ]

    _run_subprocess(prebuild_cmd, cwd=prebuild_dir, env=env)


def _generate_build_tree(cmake_path: Path,
                         source_dir: Path,
                         build_dir: Path,
                         configs: set[str],
                         args,
                         cmake_extra_args: list[str]
                         ):
    log.info("Generating CMake build tree")

    cmake_args = [
        str(cmake_path),
        str(source_dir),
        # Define Python_EXECUTABLE so find_package(python3 ...) will use the same version of python being used to
        # run this script
        "-DPython_EXECUTABLE=" + sys.executable,
        "-DOCOS_ENABLE_SELECTED_OPLIST=" + ("ON" if _is_reduced_ops_build(args) else "OFF"),
    ]

    if args.onnxruntime_version:
        cmake_args.append(f"-DOCOS_ONNXRUNTIME_VERSION={args.onnxruntime_version}")

    if args.enable_cxx_tests:
        cmake_args.append("-DOCOS_ENABLE_CTEST=ON")
        ort_lib_dir = _validate_cxx_test_args(args)
        if ort_lib_dir:
            cmake_args.append(f"-DONNXRUNTIME_LIB_DIR={str(ort_lib_dir)}")

    if args.android:
        cmake_args += [
            "-DOCOS_BUILD_ANDROID=ON",
            "-DCMAKE_TOOLCHAIN_FILE="
            + str((args.android_ndk_path / "build" / "cmake" / "android.toolchain.cmake").resolve(strict=True)),
            f"-DANDROID_PLATFORM=android-{args.android_api}",
            f"-DANDROID_ABI={args.android_abi}",
        ]

    if is_macOS() and not args.android:
        # these cmake definitions apply to MacOS and iOS builds
        cmake_args += (
            [f"-DCMAKE_OSX_ARCHITECTURES={args.apple_arch}"] +
            ([f"-DCMAKE_OSX_DEPLOYMENT_TARGET={args.apple_deploy_target}"] if args.apple_deploy_target else []) +
            (["-DOCOS_BUILD_APPLE_FRAMEWORK=ON"] if args.build_apple_framework else [])
        )

        if args.ios:
            required_args = [
                args.ios_sysroot,
                args.apple_deploy_target,
            ]

            arg_names = [
                "--ios_sysroot          " + "<the location or name of the macOS platform SDK>",
                "--apple_deploy_target  " + "<the minimum version of the target platform>",
            ]

            if not all(required_args):
                raise UsageError("iOS build is missing required arguments: "
                                + ", ".join(val for val, cond in zip(arg_names, required_args) if not cond))

            cmake_args += [
                "-DCMAKE_SYSTEM_NAME=iOS",
                "-DCMAKE_OSX_SYSROOT=" + args.ios_sysroot,
                "-DCMAKE_TOOLCHAIN_FILE=" + str(args.ios_toolchain_file.resolve(strict=True)),
            ]

    if args.wasm:
        emsdk_toolchain = (args.emsdk_path / "upstream" / "emscripten" / "cmake" / "Modules" / "Platform" /
                          "Emscripten.cmake").resolve()
        if not emsdk_toolchain.exists():
            raise UsageError(f"Emscripten toolchain file was not found at {str(emsdk_toolchain)}")

        # some things aren't currently supported with wasm so disable
        # TODO: Might be cleaner to do a selected ops build and enable/disable things via that.
        #       For now replicating the config from .az/mshost.yaml for the WebAssembly job.
        cmake_args += [
            "-DCMAKE_TOOLCHAIN_FILE=" + str(emsdk_toolchain),
            "-DOCOS_ENABLE_SPM_TOKENIZER=ON",
            "-DOCOS_BUILD_PYTHON=OFF",
            "-DOCOS_ENABLE_CV2=OFF",
            "-DOCOS_ENABLE_VISION=OFF"
        ]

    if args.disable_exceptions:
        cmake_args.append("-DOCOS_ENABLE_CPP_EXCEPTIONS=OFF")

    if args.build_java:
        cmake_args.append("-DOCOS_BUILD_JAVA=ON")

    cmake_args += ["-D{}".format(define) for define in args.cmake_extra_defines]
    cmake_args += cmake_extra_args

    for config in configs:
        config_build_dir = _get_build_config_dir(build_dir, config)
        _run_subprocess(cmake_args + [f"-DCMAKE_BUILD_TYPE={config}"], cwd=config_build_dir)


def clean_targets(cmake_path: Path, build_dir: Path, configs: set[str]):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = _get_build_config_dir(build_dir, config)
        cmd_args = [str(cmake_path), "--build", str(build_dir2), "--config", config, "--target", "clean"]

        _run_subprocess(cmd_args)


def build_targets(args, cmake_path: Path, build_dir: Path, configs: set[str], num_parallel_jobs: int):
    env = {}
    if args.android:
        env["ANDROID_HOME"] = str(args.android_home)
        env["ANDROID_NDK_HOME"] = str(args.android_ndk_path)

    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = _get_build_config_dir(build_dir, config)
        cmd_args = [str(cmake_path), "--build", str(build_dir2), "--config", config]

        build_tool_args = []
        if num_parallel_jobs != 1:
            if args.cmake_generator.startswith("Visual Studio"):
                build_tool_args += [
                    "/maxcpucount:{}".format(num_parallel_jobs),
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                ]
            elif args.cmake_generator in ("Ninja", "Unix Makefiles"):
                build_tool_args += ["-j{}".format(num_parallel_jobs)]
            else:
                cmd_args += ["--parallel", str(num_parallel_jobs)]

        if build_tool_args:
            cmd_args += ["--"]
            cmd_args += build_tool_args

        _run_subprocess(cmd_args, env=env)


def _run_python_tests():
    # TODO: Run the python tests in /python
    pass


def _run_android_tests(args, config: str, config_build_dir: Path):
    source_dir = REPO_DIR
    sdk_tools = android.get_sdk_tool_paths(str(args.android_home))
    adb_global_options = ["-s", args.android_adb_device_serial] if args.android_adb_device_serial is not None else []

    def adb_push(host_src: Path, device_dest: PurePosixPath, **kwargs):
        return _run_subprocess(
            [sdk_tools.adb] + adb_global_options + ["push", str(host_src), str(device_dest)], **kwargs
        )

    def adb_shell(*args, **kwargs):
        return _run_subprocess([sdk_tools.adb] + adb_global_options + ["shell", *args], **kwargs)

    device_abi_list = adb_shell("getprop ro.product.cpu.abilist", capture_stdout=True).stdout.decode().strip()
    device_preferred_abi = device_abi_list.split(sep=",", maxsplit=1)[0]

    if device_preferred_abi != args.android_abi:
        log.warning(f"Skipping Android tests because the device/emulator preferred ABI ({device_preferred_abi}) does "
                    f"not match the target Android ABI ({args.android_abi}).")
        return

    device_dir = PurePosixPath(f"/data/local/tmp/onnxruntime_extensions/{config}")
    adb_shell(f'rm -rf "{device_dir}" && mkdir -p "{device_dir}"')

    # copy shared libraries
    adb_push(config_build_dir / "bin" / "libortextensions.so", device_dir / "libortextensions.so")
    adb_push(config_build_dir / "bin" / "libonnxruntime.so", device_dir / "libonnxruntime.so")

    # copy test data
    adb_push(source_dir / "test" / "data", device_dir / "data")

    # copy and run test programs
    for test_program_name in ["extensions_test", "ocos_test"]:
        device_test_program_path = device_dir / test_program_name
        adb_push(config_build_dir / "bin" / test_program_name, device_test_program_path)
        adb_shell(f'chmod 755 "{device_test_program_path}"')
        adb_shell(f'cd "{device_dir}" && '
                  f'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:{device_dir}" "{device_test_program_path}"')


def _run_ios_tests(args, config: str, config_build_dir: Path):
    source_dir = REPO_DIR

    simulator_device_info = _run_subprocess(
        [
            sys.executable,
            str(source_dir / "tools" / "ios" / "get_simulator_device_info.py")
         ],
         capture_stdout=True).stdout.decode().strip()

    log.debug(f"Simulator device info:\n{simulator_device_info}")

    simulator_device_info = json.loads(simulator_device_info)

    # run test schemes
    for scheme_name in ["ocos_test", "extensions_test"]:
        _run_subprocess(
            [
                "xcrun",
                "xcodebuild",
                "test-without-building",
                "-project",
                str(config_build_dir / "onnxruntime_extensions.xcodeproj"),
                "-configuration",
                config,
                "-scheme",
                scheme_name,
                "-destination",
                f"platform=iOS Simulator,id={simulator_device_info['device_udid']}",
            ])


def _run_cxx_tests(args, build_dir: Path, configs: set[str]):
    code_coverage_using_vstest = is_windows() and args.cxx_code_coverage
    ctest_path = _resolve_executable_path(args.ctest_path, resolution_failure_allowed=code_coverage_using_vstest)

    for config in configs:
        log.info("Running tests for %s configuration", config)

        config_build_dir = _get_build_config_dir(build_dir, config)

        if args.android:
            _run_android_tests(args, config, config_build_dir)
            continue
        elif args.ios:
            _run_ios_tests(args, config, config_build_dir)
            continue

        if code_coverage_using_vstest:
            # Get the "Google Test Adapter" for vstest.
            if not (config_build_dir / "GoogleTestAdapter.0.18.0").is_dir():
                _run_subprocess(
                    [
                        "nuget.exe",
                        "restore",
                        str(REPO_DIR / "test" / "packages.config"),
                        "-ConfigFile",
                        str(REPO_DIR / "test" / "NuGet.config"),
                        "-PackagesDirectory",
                        str(config_build_dir),
                    ]
                )

            # test exes are in the bin/<config> subdirectory of the build output dir
            # call resolve() to get the full path as we're going to execute in build_dir not config_build_dir
            test_dir = (config_build_dir / "bin" / config).resolve()
            adapter = (config_build_dir / 'GoogleTestAdapter.0.18.0' / 'build' / '_common').resolve()

            executables = [
                str(test_dir / "extensions_test.exe"),
                str(test_dir / "ocos_test.exe")
            ]

            # run this script from a VS dev shell so vstest.console.exe is found via PATH
            vstest_exe = _resolve_executable_path("vstest.console.exe")
            _run_subprocess(
                [
                    vstest_exe,
                    "--parallel",
                    f"--TestAdapterPath:{str(adapter)}",
                    "/Logger:trx",
                    "/Enablecodecoverage",
                    "/Platform:x64",
                    f"/Settings:{str(REPO_DIR / 'test' / 'codeconv.runsettings')}",
                ]
                + executables,
                cwd=build_dir,
            )
        else:
            ctest_cmd = [str(ctest_path), "--build-config", config, "--verbose", "--timeout", "10800"]
            _run_subprocess(ctest_cmd, cwd=config_build_dir)


def main():
    log.debug("Command line arguments:\n  {}".format(" ".join(shlex.quote(arg) for arg in sys.argv[1:])))

    args = _parse_arguments()
    cross_compiling = any((args.arm, args.arm64, args.arm64ec,
                           args.android,
                           args.wasm,
                           _is_cross_compiling_on_apple(args)))

    # If there was no explicit argument saying what to do, default
    # to update, build and test (for native builds).
    if not (args.update or args.clean or args.build or args.test):
        log.debug("Defaulting to running update, build [and test for native builds].")
        args.update = True
        args.build = True
        args.test = not cross_compiling

    if args.skip_tests:
        args.test = False

    if args.android:
        original_cmake_generator = args.cmake_generator
        if original_cmake_generator not in ["Ninja", "Unix Makefiles"]:
            if _resolve_executable_path("ninja", resolution_failure_allowed=True) is not None:
                args.cmake_generator = "Ninja"
            elif _resolve_executable_path("make", resolution_failure_allowed=True) is not None:
                args.cmake_generator = "Unix Makefiles"
            else:
                raise UsageError("Unable to find appropriate CMake generator for cross-compiling for Android. "
                                 "Valid generators are 'Ninja' or 'Unix Makefiles'.")

        if args.cmake_generator != original_cmake_generator:
            log.info(f"Setting CMake generator to '{args.cmake_generator}' for cross-compiling for Android.")

    if args.ios:
        if args.cmake_generator != "Xcode":
            args.cmake_generator = "Xcode"
            log.info(f"Setting CMake generator to 'Xcode' for cross-compiling for iOS.")

    configs = set(args.config)

    # setup paths and directories
    try:
        cmake_path = _resolve_executable_path(args.cmake_path, resolution_failure_allowed=False)
    except ValueError as e:
        raise UsageError("Unable to find CMake executable. Please specify its path with --cmake_path.") from e

    build_dir = args.build_dir

    if args.update or args.build:
        for config in configs:
            os.makedirs(_get_build_config_dir(build_dir, config), exist_ok=True)

    if args.wasm:
        _setup_emscripten(args)

    log.info("Build started")

    if args.update:
        if args.android:
            _android_prebuild(args.android_abi, args.android_ndk_path, args.android_api)

        if _is_reduced_ops_build(args):
            log.info("Generating config for selected ops")
            _generate_selected_ops_config(args.include_ops_by_config)

        cmake_extra_args = []

        if is_windows():
            cpu_arch = platform.architecture()[0]
            if args.wasm:
                cmake_extra_args = ["-G", "Ninja"]
            elif args.cmake_generator == "Ninja":
                if cpu_arch == "32bit" or args.arm or args.arm64 or args.arm64ec:
                    raise UsageError(
                        "To cross-compile with Ninja, load the toolset environment for the target processor "
                        "(e.g. Cross Tools Command Prompt for VS)")
                cmake_extra_args = ["-G", args.cmake_generator]
            elif args.arm or args.arm64 or args.arm64ec:
                # Cross-compiling for ARM(64) architecture
                if args.arm:
                    cmake_extra_args = ["-A", "ARM"]
                elif args.arm64:
                    cmake_extra_args = ["-A", "ARM64"]
                elif args.arm64ec:
                    cmake_extra_args = ["-A", "ARM64EC"]

                cmake_extra_args += ["-G", args.cmake_generator]

                # Cannot test on host build machine for cross-compiled
                # builds (Override any user-defined behaviour for test if any)
                if args.test:
                    log.warning("Cannot test on host build machine for cross-compiled ARM(64) builds. "
                                "Will skip test running after build.")
                    args.test = False
            elif cpu_arch == "32bit" or args.x86:
                cmake_extra_args = ["-A", "Win32", "-T", "host=x64", "-G", args.cmake_generator]
            else:
                toolset = "host=x64"
                # TODO: Do we need the ability to specify the toolset? If so need to add the msvc_toolset arg back in
                # if args.msvc_toolset:
                #     toolset += f",version={args.msvc_toolset}"
                cmake_extra_args = ["-A", "x64", "-T", toolset, "-G", args.cmake_generator]
        else:
            cmake_extra_args += ["-G", args.cmake_generator]

        if is_macOS():
            if not args.ios and not args.android and args.apple_arch == "arm64" and platform.machine() == "x86_64":
                if args.test:
                    log.warning("Cannot test ARM64 build on X86_64. Will skip test running after build.")
                    args.test = False

        _generate_build_tree(
            cmake_path,
            REPO_DIR,
            build_dir,
            configs,
            args,
            cmake_extra_args)

    if args.clean:
        clean_targets(cmake_path, build_dir, configs)

    if args.build:
        if args.parallel < 0:
            raise UsageError("Invalid parallel job count: {}".format(args.parallel))
        num_parallel_jobs = os.cpu_count() if args.parallel == 0 else args.parallel
        build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs)

    if args.test:
        _run_python_tests()
        if args.enable_cxx_tests:
            _validate_cxx_test_args(args)
            _run_cxx_tests(args, build_dir, configs)

    log.info("Build complete")


if __name__ == "__main__":
    try:
        main()
    except UsageError as e:
        log.error(str(e))
        sys.exit(1)
