#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# builds an Android AAR package for the specified ABIs

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import List

_script_dir = Path(__file__).resolve().parent
_repo_dir = _script_dir.parents[1]

sys.path.insert(0, str(_repo_dir / "tools"))

from utils import get_logger, is_windows, run

_supported_abis = ["armeabi-v7a", "arm64-v8a", "x86", "x86_64"]
_log = get_logger("build_aar")


def build_for_abi(
    build_dir: Path,
    config: str,
    abi: str,
    api_level: int,
    sdk_path: Path,
    ndk_path: Path,
    cmake_extra_defines: List[str],
):
    build_cmd = [
        sys.executable,
        str(_repo_dir / "tools" / "build.py"),
        f"--build_dir={build_dir}",
        f"--config={config}",
        "--update",
        "--build",
        "--parallel",
        "--test",
        # Android options
        "--android",
        f"--android_abi={abi}",
        f"--android_api={api_level}",
        f"--android_home={sdk_path}",
        f"--android_ndk_path={ndk_path}",
    ] + ((["--cmake_extra_defines"] + cmake_extra_defines) if cmake_extra_defines else [])

    run(*build_cmd)


def build_aar(
    output_dir: Path,
    config: str,
    build_target: str,
    abis: List[str],
    api_level: int,
    sdk_path: Path,
    ndk_path: Path,
    cmake_extra_defines: List[str],
):
    output_dir = output_dir.resolve()

    sdk_path = sdk_path.resolve(strict=True)
    assert sdk_path.is_dir()

    ndk_path = ndk_path.resolve(strict=True)
    assert ndk_path.is_dir()

    intermediates_dir = output_dir / "intermediates"
    base_jnilibs_dir = intermediates_dir / "jnilibs" / config

    if build_target in ["so", "aar"]:
        for abi in abis:
            build_dir = intermediates_dir / abi
            build_for_abi(build_dir, config, abi, api_level, sdk_path, ndk_path, cmake_extra_defines)

            # copy JNI library files to jnilibs_dir
            jnilibs_dir = base_jnilibs_dir / abi
            jnilibs_dir.mkdir(parents=True, exist_ok=True)

            jnilib_names = ["libonnxruntime_extensions4j_jni.so"]
            for jnilib_name in jnilib_names:
                shutil.copyfile(build_dir / config / "java" / "android" / abi / jnilib_name, jnilibs_dir / jnilib_name)

    # early return if only building JNI libraries
    # To accerlate the build pipeline, we can build the JNI libraries first in parallel for different abi, 
    # and then build the AAR package.
    if build_target == "so":
        return

    java_root = _repo_dir / "java"
    gradle_build_file = java_root / "build-android.gradle"
    gradle_settings_file = java_root / "settings-android.gradle"
    aar_build_dir = intermediates_dir / "aar" / config
    aar_publish_dir = output_dir / "aar_out" / config
    ndk_version = ndk_path.name  # infer NDK version from NDK path

    gradle_path = java_root / ("gradlew.bat" if is_windows() else "gradlew")
    aar_build_cmd = [
        str(gradle_path),
        "clean",
        "build",
        "publish",
        "--no-daemon",
        f"-b={gradle_build_file}",
        f"-c={gradle_settings_file}",
        f"-DjniLibsDir={base_jnilibs_dir}",
        f"-DbuildDir={aar_build_dir}",
        f"-DpublishDir={aar_publish_dir}",
        f"-DminSdkVer={api_level}",
        f"-DndkVer={ndk_version}",
    ]

    env = os.environ.copy()
    env.update({"ANDROID_HOME": str(sdk_path), "ANDROID_NDK_HOME": str(ndk_path)})

    run(*aar_build_cmd, env=env, cwd=java_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Builds the Android AAR package for onnxruntime-extensions.")

    def path_from_env_var(env_var: str):
        env_var_value = os.environ.get(env_var)
        return Path(env_var_value) if env_var_value is not None else None

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--config",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        default="Debug",
        help="CMake build configuration.",
    )

    parser.add_argument(
        "--build_target",
        type=str,
        choices=["aar", "so", "pack_aar"],
        default="aar",
        help="Path to the Android NDK. Typically `<Android SDK>/ndk/<ndk_version>`.",
    )

    parser.add_argument(
        "--abi",
        action="append",
        choices=_supported_abis,
        dest="abis",
        help="Specify a target Android Application Binary Interface (ABI). Repeat to specify multiple ABIs. "
        "If no ABIs are specified, all supported ABIs will be built.",
    )

    parser.add_argument(
        "--api-level",
        type=int,
        default=24,
        help="Android API Level. E.g., 24.",
    )

    parser.add_argument(
        "--sdk-path",
        type=Path,
        default=path_from_env_var("ANDROID_HOME"),
        help="Path to the Android SDK.",
    )

    parser.add_argument(
        "--ndk-path",
        type=Path,
        default=path_from_env_var("ANDROID_NDK_HOME"),
        help="Path to the Android NDK. Typically `<Android SDK>/ndk/<ndk_version>`.",
    )

    parser.add_argument(
        "--cmake-extra-defines",
        action="append",
        nargs="+",
        default=[],
        help="Extra definition(s) to pass to CMake (with the CMake -D option) during build system generation. "
        "E.g., `--cmake-extra-defines OPTION1=ON OPTION2=OFF --cmake-extra-defines OPTION3=ON`.",
    )

    args = parser.parse_args()

    args.abis = args.abis or _supported_abis.copy()

    assert (
        args.sdk_path is not None
    ), "Android SDK path must be provided with --sdk-path or environment variable ANDROID_HOME."

    assert (
        args.ndk_path is not None
    ), "Android NDK path must be provided with --ndk-path or environment variable ANDROID_NDK_HOME."

    # convert from List[List[str]] to List[str]
    args.cmake_extra_defines = [element for inner_list in args.cmake_extra_defines for element in inner_list]

    return args


def main():
    args = parse_args()

    _log.info(f"Building AAR for ABIs: {args.abis}")

    build_aar(
        output_dir=args.output_dir,
        config=args.config,
        build_target=args.build_target,
        abis=args.abis,
        api_level=args.api_level,
        sdk_path=args.sdk_path,
        ndk_path=args.ndk_path,
        cmake_extra_defines=args.cmake_extra_defines,
    )

    _log.info("AAR build complete.")


if __name__ == "__main__":
    main()
