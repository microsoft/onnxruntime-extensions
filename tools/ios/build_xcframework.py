#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build an xcframework from individual per-platform/arch static frameworks

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

_repo_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_dir / "tools"))

from utils import get_logger, run  # noqa

_supported_platform_archs = {
    "iphoneos": ["arm64"],
    "iphonesimulator": ["x86_64", "arm64"],
}

_lipo = "lipo"
_xcrun = "xcrun"

_log = get_logger("build_xcframework")


def _get_opencv_toolchain_file(platform: str, opencv_dir: Path):
    return (
        opencv_dir
        / "platforms/ios/cmake/Toolchains"
        / ("Toolchain-iPhoneOS_Xcode.cmake" if platform == "iphoneos" else "Toolchain-iPhoneSimulator_Xcode.cmake")
    )


def _rmtree_if_existing(dir: Path):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        pass


def build_framework_for_platform_and_arch(
    build_dir: Path,
    platform: str,
    arch: str,
    config: str,
    opencv_dir: Path,
    ios_deployment_target: str,
    other_build_args: list[str],
):
    cmake_defines = [
        # required by OpenCV CMake toolchain file
        # https://github.com/opencv/opencv/blob/4223495e6cd67011f86b8ecd9be1fa105018f3b1/platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake#L64-L66
        f"IOS_ARCH={arch}",
        # required by OpenCV CMake toolchain file
        # https://github.com/opencv/opencv/blob/4223495e6cd67011f86b8ecd9be1fa105018f3b1/platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake#L96-L101
        f"IPHONEOS_DEPLOYMENT_TARGET={ios_deployment_target}",
    ]

    build_cmd = (
        [
            sys.executable,
            str(_repo_dir / "tools" / "build.py"),
            f"--build_dir={build_dir}",
            f"--config={config}",
            "--update",
            "--build",
            "--parallel",
            "--test",
            # iOS options
            "--ios",
            "--build_apple_framework",
            f"--ios_sysroot={platform}",
            f"--ios_toolchain_file={_get_opencv_toolchain_file(platform, opencv_dir)}",
            f"--apple_arch={arch}",
            f"--apple_deploy_target={ios_deployment_target}",
        ]
        + [f"--one_cmake_extra_define={cmake_define}" for cmake_define in cmake_defines]
        + other_build_args
    )

    run(*build_cmd)


def build_xcframework(
    output_dir: Path,
    platform_archs: dict[str, list[str]],
    mode: str,
    config: str,
    opencv_dir: Path,
    ios_deployment_target: str,
    other_build_args: list[str],
):
    output_dir = output_dir.resolve()
    intermediate_build_dir = output_dir / "intermediates"
    intermediate_build_dir.mkdir(parents=True, exist_ok=True)

    assert len(platform_archs) > 0, "no platforms specified"

    for platform, archs in platform_archs.items():
        assert len(archs) > 0, f"no arch specified for platform {platform}"

    def platform_arch_framework_build_dir(platform, arch):
        return intermediate_build_dir / f"{platform}/{arch}"

    build_platform_arch_frameworks = mode in ["build_platform_arch_frameworks_only", "build_xcframework"]

    if build_platform_arch_frameworks:
        for platform, archs in platform_archs.items():
            for arch in archs:
                build_framework_for_platform_and_arch(
                    platform_arch_framework_build_dir(platform, arch),
                    platform,
                    arch,
                    config,
                    opencv_dir,
                    ios_deployment_target,
                    other_build_args,
                )

    pack_xcframework = mode in ["pack_xcframework_only", "build_xcframework"]

    if pack_xcframework:
        # the public headers and framework_info.json should be the same across platform/arch builds
        # select them from one of the platform/arch build directories to copy to the output directory
        headers_dir = None
        framework_info_file = None

        # create per-platform fat framework from platform/arch frameworks
        platform_fat_framework_dirs = []
        for platform, archs in platform_archs.items():
            arch_framework_dirs = [
                platform_arch_framework_build_dir(platform, arch)
                / config
                / "static_framework"
                / "onnxruntime_extensions.framework"
                for arch in archs
            ]

            if not build_platform_arch_frameworks:
                # if they weren't just built, check that the expected platform/arch framework directories are present
                for arch_framework_dir in arch_framework_dirs:
                    assert (
                        arch_framework_dir.is_dir()
                    ), f"platform/arch framework directory not found: {arch_framework_dir}"

            first_arch_framework_dir = arch_framework_dirs[0]

            if headers_dir is None:
                headers_dir = first_arch_framework_dir / "Headers"
                framework_info_file = first_arch_framework_dir.parents[1] / "framework_info.json"

            platform_fat_framework_dir = intermediate_build_dir / f"{platform}/onnxruntime_extensions.framework"
            _rmtree_if_existing(platform_fat_framework_dir)
            platform_fat_framework_dir.mkdir()

            # copy over files from arch-specific framework to fat framework
            for framework_file_relative_path in [Path("Headers"), Path("Info.plist")]:
                src = first_arch_framework_dir / framework_file_relative_path
                dst = platform_fat_framework_dir / framework_file_relative_path
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy(src, dst)

            # combine arch-specific framework libraries
            arch_libs = [str(framework_dir / "onnxruntime_extensions") for framework_dir in arch_framework_dirs]
            run(
                *([_lipo, "-create", "-output", str(platform_fat_framework_dir / "onnxruntime_extensions")] + arch_libs)
            )

            platform_fat_framework_dirs.append(platform_fat_framework_dir)

        # create xcframework
        xcframework_dir = output_dir / "onnxruntime_extensions.xcframework"
        _rmtree_if_existing(xcframework_dir)

        create_xcframework_cmd = [_xcrun, "xcodebuild", "-create-xcframework", "-output", str(xcframework_dir)]
        for platform_fat_framework_dir in platform_fat_framework_dirs:
            create_xcframework_cmd += ["-framework", str(platform_fat_framework_dir)]
        run(*create_xcframework_cmd)

        # copy public headers
        output_headers_dir = output_dir / "Headers"
        _rmtree_if_existing(output_headers_dir)
        shutil.copytree(headers_dir, output_headers_dir)

        # copy framework_info.json
        shutil.copyfile(framework_info_file, output_dir / "framework_info.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds an iOS xcframework.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to output directory.",
    )

    # This option is used in CI pipelines to accelerate the build process,
    # We have multiple platform/archs to build for. We can build them in parallel.
    # The parallel build works like this:
    #   1. build the platform/arch frameworks in different jobs, in parallel
    #   2. download the platform/arch framework files from the previous jobs and pack them into the xcframework
    parser.add_argument(
        "--mode",
        default="build_xcframework",
        choices=["build_xcframework", "build_platform_arch_frameworks_only", "pack_xcframework_only"],
        help="Build mode. "
        "'build_xcframework' builds the whole package. "
        "'build_platform_arch_frameworks_only' builds the platform/arch frameworks only. "
        "'pack_xcframework_only' packs the xcframework from existing lib files only. "
        "Note: 'pack_xcframework_only' assumes previous invocation(s) with mode 'build_platform_arch_frameworks_only'.",
    )

    parser.add_argument(
        "--platform_arch",
        nargs=2,
        action="append",
        metavar=("PLATFORM", "ARCH"),
        dest="platform_archs",
        help="Specify a platform/arch pair to build. Repeat to specify multiple pairs. "
        "If no pairs are specified, all supported pairs will be built.",
    )

    # platform/arch framework build-related options
    parser.add_argument(
        "--config",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        default="Debug",
        help="CMake build configuration.",
    )
    parser.add_argument(
        "--ios_deployment_target",
        default="12.0",
        help="iOS deployment target.",
    )

    args, unknown_args = parser.parse_known_args()

    # convert from [[platform1, arch1], [platform1, arch2], ...] to {platform1: [arch1, arch2, ...], ...}
    def platform_archs_from_args(platform_archs_arg: list[list[str]] | None) -> dict[str, list[str]]:
        if not platform_archs_arg:
            return _supported_platform_archs.copy()

        platform_archs = {}
        for platform, arch in platform_archs_arg:
            assert (
                platform in _supported_platform_archs.keys()
            ), f"Unsupported platform: '{platform}'. Valid values are {list(_supported_platform_archs.keys())}"
            assert arch in _supported_platform_archs[platform], (
                f"Unsupported arch for platform '{platform}': '{arch}'. "
                f"Valid values are {_supported_platform_archs[platform]}"
            )

            archs = platform_archs.setdefault(platform, [])
            if arch not in archs:
                archs.append(arch)

        return platform_archs

    args.platform_archs = platform_archs_from_args(args.platform_archs)

    return args, unknown_args


def main():
    args, unknown_args = parse_args()

    _log.info(f"Building xcframework for platform archs: {args.platform_archs}")

    build_xcframework(
        output_dir=args.output_dir,
        platform_archs=args.platform_archs,
        mode=args.mode,
        config=args.config,
        opencv_dir=_repo_dir / "cmake/externals/opencv",
        ios_deployment_target=args.ios_deployment_target,
        other_build_args=unknown_args,
    )

    _log.info("xcframework build complete.")


if __name__ == "__main__":
    main()
