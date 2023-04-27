# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import xml.etree.ElementTree as ElementTree
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
def generate_file_list(nuget_artifacts_dir):
    files_list = []
    for child in nuget_artifacts_dir.iterdir():
        if not child.is_dir():
            continue

        for cpu_arch in ["x86", "x64", "arm", "arm64"]:
            if child.name == get_package_name("win", cpu_arch):
                child = child / "lib"  # noqa: PLW2901
                for child_file in child.iterdir():
                    suffixes = [".dll", ".lib"]
                    if child_file.suffix in suffixes:
                        files_list.append({'src': str(child_file), 'target': 'runtimes/win-%s/native' % cpu_arch})
        for cpu_arch in ["x86_64", "arm64"]:
            if child.name == get_package_name("osx", cpu_arch):
                child = child / "lib"  # noqa: PLW2901
                if cpu_arch == "x86_64":
                    cpu_arch = "x64"  # noqa: PLW2901
                for child_file in child.iterdir():
                    # Check if the file has digits like onnxruntime.1.8.0.dylib. We can skip such things
                    is_versioned_dylib = re.match(r".*[\.\d+]+\.dylib$", child_file.name)
                    if child_file.is_file() and child_file.suffix == ".dylib" and not is_versioned_dylib:
                        files_list.append({'src': str(child_file), 'target': 'runtimes/osx.10.14-%s/native' % cpu_arch})
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
                        files_list.append({'src': str(child_file), 'target': 'runtimes/linux-%s/native' % cpu_arch})

        if child.name == "onnxruntime-extensions-android":
            for child_file in child.iterdir():
                if child_file.suffix in [".aar"]:
                    files_list.append({'src': str(child_file), 'target': 'runtimes/android/native'})
        if child.name == "onnxruntime_extensions.xcframework":
            files_list.append({'src': str(child)+"/**", 'target': 'runtimes/ios/native'})
    return files_list

def is_windows():
    return sys.platform.startswith("win")


def is_linux():
    return sys.platform.startswith("linux")


def is_macos():
    return sys.platform.startswith("darwin")


def validate_platform():
    if not (is_windows() or is_linux() or is_macos()):
        raise Exception("Native Nuget generation is currently supported only on Windows, Linux, and MacOS")


def generate_by_existed_nuspec(args):
    template_nuspec_path = args.source_path/"nuget"/"NativeNuget.nuspec"
    output_nuspec_path = Path(args.native_build_path)/"NativeNuget.nuspec"
    tree = ElementTree.parse(template_nuspec_path)
    root = tree.getroot()

    # update version and commit id
    packages_node = root.findall('metadata')[0]
    for package_item in list(packages_node):
        if package_item.tag == "version" and args.package_version:
            package_item.text = args.package_version
        elif package_item.tag == "repository" and args.commit_id:
            package_item.attrib['commit'] = args.commit_id

    # remove file in local build
    files_node = root.findall('files')[0]
    for file_item in list(files_node):
        if 'runtimes' in file_item.attrib['target']:
            files_node.remove(file_item)

    # add files dynamically
    nuget_artifacts_dir = args.native_build_path/"nuget-artifacts-ort-ext"
    for file_item in generate_file_list(nuget_artifacts_dir):
        file_node = ElementTree.SubElement(files_node, 'file')
        file_node.attrib = file_item

    # align indent
    py_version = sys.version_info
    if py_version > (3, 9):
        ElementTree.indent(root)

    tree.write(output_nuspec_path, encoding='utf-8', xml_declaration=True)
    return


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime extensions create nuget spec script (for hosting native shared library artifacts)",
        usage="",
    )
    # Main arguments
    parser.add_argument("--package_version", default='', help="ORT package version. Eg: 1.0.0")
    parser.add_argument("--sources_path", required=True, type=Path, help="sources repo path.")
    parser.add_argument("--native_build_path", default='./nuget-artifacts',
                        type=Path, help="Native build output directory.")
    parser.add_argument("--packages_path", default='./', type=Path, help="Nuget packages output directory.")
    parser.add_argument("--commit_id", default='', help="The last commit id included in this package.")
    parser.add_argument(
        "--is_release_build",
        default='false',
        type=str,
        help="Flag indicating if the build is a release build. Accepted values: true/false.",
    )
    args = parser.parse_args()
    args.native_build_path = args.native_build_path.resolve()
    args.sources_path = args.sources_path.resolve()
    args.packages_path = args.packages_path.resolve()
    return args

def main():
    # Parse arguments
    args = parse_arguments()

    validate_platform()

    if args.is_release_build.lower() != "true" and args.is_release_build.lower() != "false":
        raise Exception("Only valid options for IsReleaseBuild are: true and false")

    generate_by_existed_nuspec(args)


if __name__ == "__main__":
    sys.exit(main())
