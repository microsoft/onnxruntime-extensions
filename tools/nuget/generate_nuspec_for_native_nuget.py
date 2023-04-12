# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
import sys
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create ORT Extensions nuget spec script", usage=""
    )
    # Main arguments
    parser.add_argument("--package_name", required=True, help="ORT package name. Eg: ort-extensions-nuget")
    parser.add_argument("--package_version", required=True, help="ORT package version. Eg: 0.7.0")
    parser.add_argument("--native_build_path", required=True, help="Native build output directory.")
    parser.add_argument("--sources_path", required=True, help="Extensions source code root.")

    return parser.parse_args()

def generate_id(line_list, package_name):
    line_list.append("<id>" + package_name + "</id>")


def generate_version(line_list, package_version):
    line_list.append("<version>" + package_version + "</version>")


def generate_authors(line_list, authors):
    line_list.append("<authors>" + authors + "</authors>")


def generate_owners(line_list, owners):
    line_list.append("<owners>" + owners + "</owners>")


def generate_description(line_list, package_name):
    line_list.append("<description>ONNX Runtime Extensions NuGet Package</description>")


def generate_copyright(line_list, copyright):
    line_list.append("<copyright>" + copyright + "</copyright>")


def generate_tags(line_list, tags):
    line_list.append("<tags>" + tags + "</tags>")


def generate_license(line_list):
    line_list.append('<license type="file">LICENSE.txt</license>')


def generate_project_url(line_list, project_url):
    line_list.append("<projectUrl>" + project_url + "</projectUrl>")


def generate_repo_url(line_list, repo_url, commit_id):
    line_list.append('<repository type="git" url="' + repo_url + '"' + ' commit="' + commit_id + '" />')

def generate_dependencies(line_list):
    dependencies_list = ['<group targetFramework="native">']

    # Update dependencies as needed here:
    dependencies_list.append('<dependency id="blingfire" version="0831265c1aca95ca02eca5bf1155e4251e545328"/>')
    dependencies_list.append('<dependency id="dlib" version="19.22"/>')
    dependencies_list.append('<dependency id="google/re2" version="2020-11-01"/>')
    dependencies_list.append('<dependency id="nlohmann/json" version="3.7.3"/>')
    dependencies_list.append('<dependency id="opencv" version="4.5.4"/>')
    dependencies_list.append('<dependency id="sentencepiece" version="0.1.96"/>')

    dependencies_list.append("</group>")
    line_list.append("<dependencies>")
    line_list.append(dependencies_list)
    line_list.append("</dependencies>")

def generate_release_notes(line_list, package_version):
    line_list.append("<releaseNotes>")

    # Update release notes as needed here:
    if package_version == "0.7.0":
        line_list.append('''
        General
            1. New custom operators: RobertaTokenizer, ClipTokenizer, EncodeImage, DecodeImage
            2. ORT custom operator C++ stub generation tool
            3. Operator implementation and documentation improved.
            4. Python (3.7 - 3.10) and ORT (1.10 above) compatible.

        Mobile
            1. Android package: Maven
            2. iOS package: CocoaPods
            3. PrePostProcessor tool for mobile model
            4. Super-resolution model pre- / post- processing end-to-end examples''')
    
    line_list.append("</releaseNotes>")

def generate_metadata(line_list, args):
    metadata_list = ["<metadata>"]
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, "Microsoft")
    generate_owners(metadata_list, "Microsoft")
    generate_description(metadata_list, args.package_name)
    generate_copyright(metadata_list, "Copyright (c) Microsoft Corporation. All rights reserved.")
    generate_tags(metadata_list, "ONNX Runtime Extensions")
    generate_license(metadata_list)
    generate_project_url(metadata_list, "https://github.com/microsoft/onnxruntime-extensions")
    generate_dependencies(metadata_list)
    generate_release_notes(metadata_list, args.package_version)
    metadata_list.append("</metadata>")

    line_list += metadata_list

def generate_files(line_list, args):
    files_list = ["<files>"]

    # TODO: Add all project files for NuGet package here
    files_list.append(
        "<file src=" + '"' + os.path.join(args.sources_path, "README.md") + '" target="native" />'
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

def main():
    # Parse arguments
    args = parse_arguments()

    # Generate nuspec
    lines = generate_nuspec(args)

    # Create the nuspec needed to generate the Nuget
    with open(os.path.join(args.native_build_path, "OrtExtNativeNuget.nuspec"), "w") as f:
        for line in lines:
            # Uncomment the printing of the line if you need to debug what's produced on a CI machine
            # print(line)
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    sys.exit(main())