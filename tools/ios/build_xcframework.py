#!/usr/bin/env python3

# build an xcframework from individual per-platform/arch static frameworks

import argparse
import pathlib
import shutil
import subprocess
import typing

script_dir = pathlib.Path(__file__).resolve()
repo_dir = script_dir.parents[2]

default_platform_archs = {
    "iphoneos": ["arm64"],
    "iphonesimulator": ["x86_64", "arm64"],
}

cmake = "cmake"
lipo = "lipo"
xcrun = "xcrun"


def _get_opencv_toolchain_file(platform: str, opencv_dir: pathlib.Path):
    return (
        opencv_dir
        / "platforms/ios/cmake/Toolchains"
        / ("Toolchain-iPhoneOS_Xcode.cmake" if platform == "iphoneos" else "Toolchain-iPhoneSimulator_Xcode.cmake")
    )


def _run(cmd_args: typing.List[str], **kwargs):
    import shlex

    print(f"Running command:\n  {shlex.join(cmd_args)}")
    subprocess.run(cmd_args, check=True, **kwargs)


def _rmtree_if_existing(dir: pathlib.Path):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        pass


def build_framework_for_platform_and_arch(
    build_dir: pathlib.Path, platform: str, arch: str, config: str, opencv_dir: pathlib.Path, ios_deployment_target: str
) -> pathlib.Path:
    build_dir.mkdir(parents=True, exist_ok=True)

    # generate build files
    generate_args = [
        cmake,
        "-G=Xcode",
        f"-S={repo_dir}",
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
        # OpenCV CMake config
        # tell OpenCV to build zlib so we can link to the static library
        "-DBUILD_ZLIB=ON",
        # use OpenCV's CMake toolchain files
        f"-DCMAKE_TOOLCHAIN_FILE={_get_opencv_toolchain_file(platform, opencv_dir)}",
        "-DCPU_BASELINE=DETECT",
        f"-DIOS_ARCH={arch}",
        f"-DIPHONEOS_DEPLOYMENT_TARGET={ios_deployment_target}",
    ]
    _run(generate_args)

    # build
    _run([cmake, f"--build", f"{build_dir}", f"--config={config}", "--parallel"])

    return build_dir / "static_framework/onnxruntime_extensions.framework"


def build_xcframework(
    output_dir: pathlib.Path,
    platform_archs: typing.Dict[str, typing.List[str]],
    config: str,
    opencv_dir: pathlib.Path,
    ios_deployment_target: str,
):
    intermediate_build_dir = output_dir / "intermediates"
    intermediate_build_dir.mkdir(parents=True, exist_ok=True)

    assert len(platform_archs) > 0, "no platforms specified"

    platform_fat_framework_dirs = []
    for platform, archs in platform_archs.items():
        assert len(archs) > 0, f"no arch specified for platform {platform}"
        arch_framework_dirs = []
        for arch in archs:
            arch_framework_dir = build_framework_for_platform_and_arch(
                intermediate_build_dir / f"{platform}/{arch}", platform, arch, config, opencv_dir, ios_deployment_target
            )

            arch_framework_dirs.append(arch_framework_dir)

        platform_fat_framework_dir = intermediate_build_dir / f"{platform}/onnxruntime_extensions.framework"
        _rmtree_if_existing(platform_fat_framework_dir)
        platform_fat_framework_dir.mkdir()

        # copy over files from arch-specific framework
        for framework_file_relative_path in [pathlib.Path("Headers"), pathlib.Path("Info.plist")]:
            src = arch_framework_dirs[0] / framework_file_relative_path
            dst = platform_fat_framework_dir / framework_file_relative_path
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

        # combine arch-specific framework libraries
        arch_libs = [str(framework_dir / "onnxruntime_extensions") for framework_dir in arch_framework_dirs]
        _run([lipo, "-create", "-output", str(platform_fat_framework_dir / "onnxruntime_extensions")] + arch_libs)

        platform_fat_framework_dirs.append(platform_fat_framework_dir)

    # create xcframework
    xcframework_dir = output_dir / "onnxruntime_extensions.xcframework"
    _rmtree_if_existing(xcframework_dir)

    create_xcframework_args = [xcrun, "xcodebuild", "-create-xcframework", "-output", str(xcframework_dir)]
    for platform_fat_framework_dir in platform_fat_framework_dirs:
        create_xcframework_args += ["-framework", str(platform_fat_framework_dir)]
    _run(create_xcframework_args)

    # copy headers
    # framework header dirs are all the same, pick one
    framework_header_dir = platform_fat_framework_dirs[0] / "Headers"
    output_header_dir = output_dir / "Headers"
    _rmtree_if_existing(output_header_dir)
    shutil.copytree(framework_header_dir, output_header_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Helper script to build an iOS xcframework.")

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
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

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir.resolve()

    build_xcframework(
        output_dir=output_dir,
        platform_archs=default_platform_archs,
        config=args.config,
        opencv_dir=repo_dir / "cmake/externals/opencv",
        ios_deployment_target=args.ios_deployment_target,
    )


if __name__ == "__main__":
    main()
