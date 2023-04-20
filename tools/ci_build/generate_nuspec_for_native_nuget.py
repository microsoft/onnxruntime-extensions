# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
import sys
from pathlib import Path


# What does the names of our C API tarball/zip files looks like
# os: win, linux, osx
def get_package_name(os, cpu_arch):
    pkg_name = None
    if os == "win":
        pkg_name = "onnxruntime-extensions-win-"
        pkg_name += cpu_arch
    elif os == "linux":
        pkg_name = "onnxruntime-extensions-linux-"
        pkg_name += cpu_arch
    elif os == "osx":
        pkg_name = "onnxruntime-extensions-osx-" + cpu_arch
    return pkg_name


# nuget_artifacts_dir: the directory with uncompressed C API tarball/zip files
# files_list: a list of xml string pieces to append
# This function has no return value. It updates files_list directly
def generate_file_list(nuget_artifacts_dir, files_list, include_pdbs):
    for child in nuget_artifacts_dir.iterdir():
        if not child.is_dir():
            continue

        for cpu_arch in ["x86", "x64", "arm", "arm64"]:
            if child.name == get_package_name("win", cpu_arch):
                child = child / "lib"  # noqa: PLW2901
                for child_file in child.iterdir():
                    suffixes = [".dll", ".lib", ".pdb"] if include_pdbs else [".dll", ".lib"]
                    if child_file.suffix in suffixes:
                        files_list.append(
                            '<file src="' + str(child_file) + '" target="runtimes/win-%s/native"/>' % cpu_arch
                        )
        for cpu_arch in ["x86_64", "arm64"]:
            if child.name == get_package_name("osx", cpu_arch):
                child = child / "lib"  # noqa: PLW2901
                if cpu_arch == "x86_64":
                    cpu_arch = "x64"  # noqa: PLW2901
                for child_file in child.iterdir():
                    # Check if the file has digits like onnxruntime.1.8.0.dylib. We can skip such things
                    is_versioned_dylib = re.match(r".*[\.\d+]+\.dylib$", child_file.name)
                    if child_file.is_file() and child_file.suffix == ".dylib" and not is_versioned_dylib:
                        files_list.append(
                            '<file src="' + str(child_file) + '" target="runtimes/osx.10.14-%s/native"/>' % cpu_arch
                        )
        for cpu_arch in ["x64", "aarch64"]:
            if child.name == get_package_name("linux", cpu_arch):
                child = child / "lib"  # noqa: PLW2901
                if cpu_arch == "x86_64":
                    cpu_arch = "x64"  # noqa: PLW2901
                elif cpu_arch == "aarch64":
                    cpu_arch = "arm64"  # noqa: PLW2901
                for child_file in child.iterdir():
                    if not child_file.is_file():
                        continue
                    if child_file.suffix == ".so":
                        files_list.append(
                            '<file src="' + str(child_file) + '" target="runtimes/linux-%s/native"/>' % cpu_arch
                        )

        if child.name == "onnxruntime-extensions-android":
            for child_file in child.iterdir():
                if child_file.suffix in [".aar"]:
                    files_list.append('<file src="' + str(child_file) + '" target="runtimes/android/native"/>')

        if child.name == "onnxruntime_extensions.xcframework":
            files_list.append('<file src="' + str(child) + "\\**" '" target="runtimes/ios/native"/>')  # noqa: ISC001


def generate_id(line_list, package_name):
    line_list.append("<id>" + package_name + "</id>")


def generate_version(line_list, package_version):
    line_list.append("<version>" + package_version + "</version>")


def generate_authors(line_list, authors):
    line_list.append("<authors>" + authors + "</authors>")


def generate_owners(line_list, owners):
    line_list.append("<owners>" + owners + "</owners>")


def generate_description(line_list, package_name):
    description = ""

    if package_name == "Microsoft.AI.MachineLearning":
        description = "This package contains Windows ML binaries."
    elif "Microsoft.ML.OnnxRuntime.Extensions" in package_name:  # This is a Microsoft.ML.OnnxRuntime.* package
        description = (
            "This package contains native shared library artifacts for all supported platforms of ONNXRuntime extensions."
        )

    line_list.append("<description>" + description + "</description>")


def generate_copyright(line_list, copyright):
    line_list.append("<copyright>" + copyright + "</copyright>")


def generate_tags(line_list, tags):
    line_list.append("<tags>" + tags + "</tags>")


def generate_icon(line_list, icon_file):
    line_list.append("<icon>" + icon_file + "</icon>")


def generate_license(line_list):
    line_list.append('<license type="file">LICENSE.txt</license>')


def generate_project_url(line_list, project_url):
    line_list.append("<projectUrl>" + project_url + "</projectUrl>")


def generate_repo_url(line_list, repo_url, commit_id):
    line_list.append('<repository type="git" url="' + repo_url + '"' + ' commit="' + commit_id + '" />')


def get_env_var(key):
    return os.environ.get(key)


def generate_release_notes(line_list, dependency_sdk_info):
    line_list.append("<releaseNotes>")
    line_list.append("""Release Def:
    General
		1. New custom operators: Whisper, DrawBoundingBoxes, RobertaTokenizer, ClipTokenizer, EncodeImage, DecodeImage
		2. Optional input/output support
		3. ORT custom operator C++ stub generation tool
		4. Operator implementation and documentation improved.
    """)

    branch = get_env_var("BUILD_SOURCEBRANCH")
    line_list.append("\t" + "Branch: " + (branch if branch is not None else ""))

    version = get_env_var("BUILD_SOURCEVERSION")
    line_list.append("\t" + "Commit: " + (version if version is not None else ""))

    build_id = get_env_var("BUILD_BUILDID")
    line_list.append(
        "\t"
        + "Build: https://aiinfra.visualstudio.com/Lotus/_build/results?buildId="
        + (build_id if build_id is not None else "")
    )

    if dependency_sdk_info:
        line_list.append("Dependency SDK: " + dependency_sdk_info)

    line_list.append("</releaseNotes>")


def generate_metadata(line_list, args):
    metadata_list = ["<metadata>"]
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, "Microsoft")
    generate_owners(metadata_list, "Microsoft")
    generate_description(metadata_list, args.package_name)
    generate_copyright(metadata_list, "\xc2\xa9 " + "Microsoft Corporation. All rights reserved.")
    generate_tags(metadata_list, "ONNXRuntime-Extensions Machine Learning")
    generate_icon(metadata_list, "ORT_icon_for_light_bg.png")
    generate_license(metadata_list)
    generate_project_url(metadata_list, "https://github.com/Microsoft/OnnxRuntime-extensions")
    generate_repo_url(metadata_list, "https://github.com/Microsoft/OnnxRuntime-extensions.git", args.commit_id)
    generate_release_notes(metadata_list, None)
    metadata_list.append("</metadata>")

    line_list += metadata_list


def generate_files(line_list, args):
    files_list = ["<files>"]

    is_cpu_package = args.package_name in ["Microsoft.ML.OnnxRuntime.Extensions"]

    is_windows_build = is_windows()

    nuget_dependencies = {}

    if is_windows_build:
        copy_command = "copy"
        runtimes_target = '" target="runtimes\\win-'
    else:
        copy_command = "cp"
        runtimes_target = '" target="runtimes\\linux-'

    runtimes_native_folder = "native"

    runtimes = f'{runtimes_target}{args.target_architecture}\\{runtimes_native_folder}"'

    # Process headers
    files_list.append(
        "<file src="
        + '"'
        + os.path.join(args.sources_path, "includes\\*.h")
        + '" target="build\\native\\include" />'
    )


    is_ado_packaging_build = False
    # Process runtimes
    # Process onnxruntime-extensions import lib, dll, and pdb
    # for android build
    if is_windows_build:
        nuget_artifacts_dir = Path(args.native_build_path) / "nuget-artifacts"
        print("nuget_artifacts_dir: ", nuget_artifacts_dir,nuget_artifacts_dir.exists())
        if nuget_artifacts_dir.exists():
            # Code path for ADO build pipeline, the files under 'nuget-artifacts' are
            # downloaded from other build jobs

            generate_file_list(nuget_artifacts_dir, files_list, False)
            is_ado_packaging_build = True
        else:
            # Code path for local dev build
            files_list.append(
                "<file src=" + '"' + os.path.join(args.native_build_path, "ortextensions.lib") + runtimes + " />"
            )
            files_list.append(
                "<file src=" + '"' + os.path.join(args.native_build_path, "ortextensions.dll") + runtimes + " />"
            )

    else:
        files_list.append(
            "<file src="
            + '"'
            + os.path.join(args.native_build_path, "nuget-staging/usr/local/lib", "libortextensions.so")
            + '" target="runtimes/linux-'+ args.target_architecture 
            + '/native" />'
        )

    if not is_cpu_package:
        # Process props file
        source_props = os.path.join(
            args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "netstandard", "props.xml"
        )
        target_props = os.path.join(
            args.sources_path,
            "csharp",
            "src",
            "Microsoft.ML.OnnxRuntime",
            "targets",
            "netstandard",
            args.package_name + ".props",
        )
        os.system(copy_command + " " + source_props + " " + target_props)
        files_list.append("<file src=" + '"' + target_props + '" target="build\\native" />')
        files_list.append("<file src=" + '"' + target_props + '" target="build\\netstandard1.1" />')
        files_list.append("<file src=" + '"' + target_props + '" target="build\\netstandard2.0" />')

        # Process targets file
        source_targets = os.path.join(
            args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "netstandard", "targets.xml"
        )
        target_targets = os.path.join(
            args.sources_path,
            "csharp",
            "src",
            "Microsoft.ML.OnnxRuntime",
            "targets",
            "netstandard",
            args.package_name + ".targets",
        )
        os.system(copy_command + " " + source_targets + " " + target_targets)
        files_list.append("<file src=" + '"' + target_targets + '" target="build\\native" />')
        files_list.append("<file src=" + '"' + target_targets + '" target="build\\netstandard1.1" />')
        files_list.append("<file src=" + '"' + target_targets + '" target="build\\netstandard2.0" />')

        # Process xamarin targets files
        if args.package_name == "Microsoft.ML.OnnxRuntime.Extensions_xmamarin":
            monoandroid_source_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "monoandroid11.0",
                "targets.xml",
            )
            monoandroid_target_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "monoandroid11.0",
                args.package_name + ".targets",
            )

            xamarinios_source_targets = os.path.join(
                args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "xamarinios10", "targets.xml"
            )
            xamarinios_target_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "xamarinios10",
                args.package_name + ".targets",
            )

            net6_android_source_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "net6.0-android",
                "targets.xml",
            )
            net6_android_target_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "net6.0-android",
                args.package_name + ".targets",
            )

            net6_ios_source_targets = os.path.join(
                args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "net6.0-ios", "targets.xml"
            )
            net6_ios_target_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "net6.0-ios",
                args.package_name + ".targets",
            )

            net6_macos_source_targets = os.path.join(
                args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "net6.0-macos", "targets.xml"
            )
            net6_macos_target_targets = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "net6.0-macos",
                args.package_name + ".targets",
            )

            os.system(copy_command + " " + monoandroid_source_targets + " " + monoandroid_target_targets)
            os.system(copy_command + " " + xamarinios_source_targets + " " + xamarinios_target_targets)
            os.system(copy_command + " " + net6_android_source_targets + " " + net6_android_target_targets)
            os.system(copy_command + " " + net6_ios_source_targets + " " + net6_ios_target_targets)
            os.system(copy_command + " " + net6_macos_source_targets + " " + net6_macos_target_targets)


    # Process License, ThirdPartyNotices, Privacy
    files_list.append("<file src=" + '"' + os.path.join(args.sources_path, "nuget", "LICENSE.txt") + '" target="LICENSE.txt" />')
    files_list.append(
        "<file src="
        + '"'
        + os.path.join(args.sources_path, "ThirdPartyNotices.txt")
        + '" target="ThirdPartyNotices.txt" />'
    )
    files_list.append(
        "<file src="
        + '"'
        + os.path.join(args.sources_path, "nuget", "ORT_icon_for_light_bg.png")
        + '" target="ORT_icon_for_light_bg.png" />'
    )
    files_list.append("</files>")

    line_list += files_list


def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append("<package>")
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append("</package>")
    return lines


def is_windows():
    return sys.platform.startswith("win")


def is_linux():
    return sys.platform.startswith("linux")


def is_macos():
    return sys.platform.startswith("darwin")


def validate_platform():
    if not (is_windows() or is_linux() or is_macos()):
        raise Exception("Native Nuget generation is currently supported only on Windows, Linux, and MacOS")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime extensions create nuget spec script (for hosting native shared library artifacts)",
        usage="",
    )
    # Main arguments
    parser.add_argument("--package_name", required=False, default="Microsoft.ML.OnnxRuntime.Extensions", 
    help="ORT package name. Eg: Microsoft.ML.OnnxRuntime.Extensions")
    parser.add_argument("--package_version", required=True, help="ORT package version. Eg: 1.0.0")
    parser.add_argument("--target_architecture", required=True, help="Eg: x64")
    parser.add_argument("--build_config", required=True, help="Eg: RelWithDebInfo")
    parser.add_argument("--native_build_path", required=True, type=Path, help="Native build output directory.")
    parser.add_argument("--packages_path", required=True, type=Path, help="Nuget packages output directory.")
    parser.add_argument("--sources_path", required=True, type=Path, help="OnnxRuntime.Extensions source code root.")
    parser.add_argument("--commit_id", required=True, help="The last commit id included in this package.")
    parser.add_argument(
        "--is_release_build",
        required=False,
        default=None,
        type=str,
        help="Flag indicating if the build is a release build. Accepted values: true/false.",
    )
    args = parser.parse_args()
    args.native_build_path = args.native_build_path.resolve()
    args.packages_path = args.packages_path.resolve()
    args.sources_path = args.sources_path.resolve()
    return args

def main():
    # Parse arguments
    args = parse_arguments()

    validate_platform()


    if args.is_release_build.lower() != "true" and args.is_release_build.lower() != "false":
        raise Exception("Only valid options for IsReleaseBuild are: true and false")

    # Generate nuspec
    lines = generate_nuspec(args)

    # Create the nuspec needed to generate the Nuget
    with open(os.path.join(args.native_build_path, "NativeNuget.nuspec"), "w") as f:
        for line in lines:
            # Uncomment the printing of the line if you need to debug what's produced on a CI machine
            # print(line)
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
