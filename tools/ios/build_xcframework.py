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

    print(f"Running command:\n  {shlex.join(cmd_args)}")
    subprocess.run(cmd_args, check=True, **kwargs)


def _rmtree_if_existing(dir: Path):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        pass


def build_framework_for_platform_and_arch(
    build_dir: Path, platform: str, arch: str, config: str, opencv_dir: Path, ios_deployment_target: str
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)

    # generate build files
    generate_args = [
        _cmake,
        "-G=Xcode",
        f"-S={_repo_dir}",
        f"-B={build_dir}",
        "-DCMAKE_SYSTEM_NAME=iOS",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={ios_deployment_target}",
        f"-DCMAKE_OSX_SYSROOT={platform}",
        f"-DCMAKE_OSX_ARCHITECTURES={arch}",
        f"-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO",
        "-DOCOS_BUILD_APPLE_FRAMEWORK=ON",
        # our version of sentencepiece doesn't support iOS build
        #   CMake Error at out/ios/RelWithDebInfo/_deps/spm-src/src/CMakeLists.txt:288 (install):
        #     install TARGETS given no BUNDLE DESTINATION for MACOSX_BUNDLE executable
        #     target "spm_encode".
        "-DOCOS_ENABLE_SPM_TOKENIZER=OFF",
        # use OpenCV's CMake toolchain file
        f"-DCMAKE_TOOLCHAIN_FILE={_get_opencv_toolchain_file(platform, opencv_dir)}",
        # required by OpenCV CMake toolchain file
        # https://github.com/opencv/opencv/blob/4223495e6cd67011f86b8ecd9be1fa105018f3b1/platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake#L64-L66
        f"-DIOS_ARCH={arch}",
        # required by OpenCV CMake toolchain file
        # https://github.com/opencv/opencv/blob/4223495e6cd67011f86b8ecd9be1fa105018f3b1/platforms/ios/cmake/Toolchains/common-ios-toolchain.cmake#L96-L101
        f"-DIPHONEOS_DEPLOYMENT_TARGET={ios_deployment_target}",
    ]
    _run(generate_args)

    # build
    _run([_cmake, f"--build", f"{build_dir}", f"--config={config}", "--parallel"])

    return build_dir / "static_framework/onnxruntime_extensions.framework"


def build_xcframework(
    output_dir: Path,
    platform_archs: Dict[str, List[str]],
    config: str,
    opencv_dir: Path,
    ios_deployment_target: str,
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

        platform_fat_framework_dirs.append(platform_fat_framework_dir)

    # create xcframework
    xcframework_dir = output_dir / "onnxruntime_extensions.xcframework"
    _rmtree_if_existing(xcframework_dir)

    create_xcframework_args = [_xcrun, "xcodebuild", "-create-xcframework", "-output", str(xcframework_dir)]
    for platform_fat_framework_dir in platform_fat_framework_dirs:
        create_xcframework_args += ["-framework", str(platform_fat_framework_dir)]
    _run(create_xcframework_args)

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
        "--ios-deployment-target",
        default="11.0",
        help="iOS deployment target.",
    )
    parser.add_argument(
        "--platform-arch",
        nargs=2,
        action="append",
        metavar=("PLATFORM", "ARCH"),
        dest="platform_archs",
        help="Specify a platform/arch pair to build. Repeat to specify multiple pairs. "
        "If no pairs are specified, all supported pairs will be built.",
    )

    args = parser.parse_args()

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

    return args


def main():
    args = parse_args()

    build_xcframework(
        output_dir=args.output_dir,
        platform_archs=args.platform_archs,
        config=args.config,
        opencv_dir=_repo_dir / "cmake/externals/opencv",
        ios_deployment_target=args.ios_deployment_target,
    )


if __name__ == "__main__":
    main()
