#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build an xcframework from individual per-platform/arch static frameworks

import argparse
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Optional

_script_dir = Path(__file__).resolve().parent
_repo_dir = _script_dir.parents[1]

_supported_platform_archs = {
    "iphoneos": ["arm64"],
    "iphonesimulator": ["x86_64", "arm64"],
}

_cmake = "cmake"
_lipo = "lipo"
_xcrun = "xcrun"


def _get_opencv_toolchain_file(platform: str, opencv_dir: Path):
    return (
        opencv_dir
        / "platforms/ios/cmake/Toolchains"
        / ("Toolchain-iPhoneOS_Xcode.cmake" if platform == "iphoneos" else "Toolchain-iPhoneSimulator_Xcode.cmake")
    )


def _run(cmd_args: List[str], **kwargs):
    import shlex

    print(f"Running command:\n  {shlex.join(cmd_args)}", flush=True)
    subprocess.run(cmd_args, check=True, **kwargs)


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
    cmake_extra_defines: List[str],
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)

    # generate build files
    generate_args = (
        [
            _cmake,
            "-G=Xcode",
            f"-S={_repo_dir}",
            f"-B={build_dir}",
        ]
        + [f"-D{cmake_extra_define}" for cmake_extra_define in cmake_extra_defines]
        + [
            "-DCMAKE_SYSTEM_NAME=iOS",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={ios_deployment_target}",
            f"-DCMAKE_OSX_SYSROOT={platform}",
            f"-DCMAKE_OSX_ARCHITECTURES={arch}",
            f"-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO",
            "-DOCOS_BUILD_APPLE_FRAMEWORK=ON",
            # use OpenCV's CMake toolchain file
            f"-DCMAKE_TOOLCHAIN_FILE={_get_opencv_toolchain_file(platform, opencv_dir)}",
            # required by OpenCV CMake toolchain file
            # https://github.com/opencv/opencv/blob/4223495e6cd67011f86b8ecd9be1fa105018f3b1/platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake#L64-L66
            f"-DIOS_ARCH={arch}",
            # required by OpenCV CMake toolchain file
            # https://github.com/opencv/opencv/blob/4223495e6cd67011f86b8ecd9be1fa105018f3b1/platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake#L96-L101
            f"-DIPHONEOS_DEPLOYMENT_TARGET={ios_deployment_target}",
        ]
    )
    _run(generate_args)

    # build
    _run([_cmake, f"--build", f"{build_dir}", f"--config={config}", "--parallel"])

    return build_dir / "static_framework/onnxruntime_extensions.framework"


def build_artifact_by_target(
    output_dir: Path,
    platform_archs: Dict[str, List[str]],
    build_target: str,
    config: str,
    opencv_dir: Path,
    ios_deployment_target: str,
    cmake_extra_defines: List[str],
):
    output_dir = output_dir.resolve()
    intermediate_build_dir = output_dir / "intermediates"
    intermediate_build_dir.mkdir(parents=True, exist_ok=True)

    assert len(platform_archs) > 0, "no platforms specified"

    # the public headers and framework_info.json should be the same across platform/arch builds
    # select them from one of the platform/arch build directories to copy to the output directory
    headers_dir = None
    framework_info_file = None

    platform_fat_framework_dirs = []
    for platform, archs in platform_archs.items():
        platform_fat_framework_dir = intermediate_build_dir / f"{platform}/onnxruntime_extensions.framework"
        platform_fat_framework_dirs.append(platform_fat_framework_dir)

    if build_target in ["lib", "xcframework"]:
        for idx, (platform, archs) in enumerate(platform_archs.items()):
            assert len(archs) > 0, f"no arch specified for platform {platform}"
            arch_framework_dirs = []
            for arch in archs:
                arch_framework_dir = build_framework_for_platform_and_arch(
                    intermediate_build_dir / f"{platform}/{arch}/{config}",
                    platform,
                    arch,
                    config,
                    opencv_dir,
                    ios_deployment_target,
                    cmake_extra_defines,
                )

                arch_framework_dirs.append(arch_framework_dir)

                if headers_dir is None:
                    headers_dir = arch_framework_dir / "Headers"
                    framework_info_file = arch_framework_dir.parents[1] / "framework_info.json"

            platform_fat_framework_dir = intermediate_build_dir / f"{platform}/onnxruntime_extensions.framework"
            _rmtree_if_existing(platform_fat_framework_dir)
            platform_fat_framework_dir.mkdir()

            # copy over files from arch-specific framework
            for framework_file_relative_path in [Path("Headers"), Path("Info.plist")]:
                src = arch_framework_dirs[0] / framework_file_relative_path
                dst = platform_fat_framework_dir / framework_file_relative_path
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy(src, dst)

            # combine arch-specific framework libraries
            arch_libs = [str(framework_dir / "onnxruntime_extensions") for framework_dir in arch_framework_dirs]
            _run([_lipo, "-create", "-output", str(platform_fat_framework_dir / "onnxruntime_extensions")] + arch_libs)

            assert platform_fat_framework_dirs[idx] == platform_fat_framework_dir

    if headers_dir:
        # copy public headers
        output_headers_dir = output_dir / "Headers"
        _rmtree_if_existing(output_headers_dir)
        shutil.copytree(headers_dir, output_headers_dir)

    if framework_info_file:
        # copy framework_info.json
        shutil.copyfile(framework_info_file, output_dir / "framework_info.json")

    # if build lib only, we're done
    if build_target == "lib":
        return

    # create xcframework
    xcframework_dir = output_dir / "onnxruntime_extensions.xcframework"
    _rmtree_if_existing(xcframework_dir)

    create_xcframework_args = [_xcrun, "xcodebuild", "-create-xcframework", "-output", str(xcframework_dir)]
    for platform_fat_framework_dir in platform_fat_framework_dirs:
        create_xcframework_args += ["-framework", str(platform_fat_framework_dir)]
    _run(create_xcframework_args)


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
    # Same as how android works
    # This one is used in the ci pipeline for accelerating the build process,
    # We have 4 archs to be build. It's in sequence by default, but we can build them in parallel.
    # The parallel build works as:
    #   1. build the lib files for each arch in different ci jobs
    #   2. download all the lib files in tasks
    #   2. pack the xcframework package in ci task
    parser.add_argument(
        "--build_target",
        default="xcframework",
        choices=["lib", "xcframework", "pack_xcframework"],
        help="Build target. 'xcframework' builds the whole package. 'lib' builds the libraries only. \
            'pack_xcframework' only does pack xcframework from existing lib files.",
    )
    parser.add_argument(
        "--config",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        default="Debug",
        help="CMake build configuration.",
    )
    parser.add_argument(
        "--ios_deployment_target",
        default="11.0",
        help="iOS deployment target.",
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
    parser.add_argument(
        "--cmake_extra_defines",
        action="append",
        nargs="+",
        default=[],
        help="Extra definition(s) to pass to CMake (with the CMake -D option) during build system generation. "
        "E.g., `--cmake_extra_defines OPTION1=ON OPTION2=OFF --cmake_extra_defines OPTION3=ON`.",
    )

    # ignore unknown args which may be present in a CI build for the nuget package as we pass through
    # all additional build flags
    args, unknown_args = parser.parse_known_args()

    # convert from [[platform1, arch1], [platform1, arch2], ...] to {platform1: [arch1, arch2, ...], ...}
    def platform_archs_from_args(platform_archs_arg: Optional[List[List[str]]]) -> Dict[str, List[str]]:
        if not platform_archs_arg:
            return _supported_platform_archs.copy()

        platform_archs = {}
        for (platform, arch) in platform_archs_arg:
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

    # convert from List[List[str]] to List[str]
    args.cmake_extra_defines = [element for inner_list in args.cmake_extra_defines for element in inner_list]

    return args


def main():
    args = parse_args()

    build_artifact_by_target(
        output_dir=args.output_dir,
        platform_archs=args.platform_archs,
        build_target=args.build_target,
        config=args.config,
        opencv_dir=_repo_dir / "cmake/externals/opencv",
        ios_deployment_target=args.ios_deployment_target,
        cmake_extra_defines=args.cmake_extra_defines,
    )


if __name__ == "__main__":
    main()
