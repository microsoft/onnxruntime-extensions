#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# assembles the iOS pod package files in a staging directory

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Dict

_script_dir = Path(__file__).resolve().parent
_repo_dir = _script_dir.parents[1]

_template_variable_pattern = re.compile(r"@(\w+)@")  # match "@var@"


def gen_file_from_template(
    template_file: Path, output_file: Path, variable_substitutions: Dict[str, str], strict: bool = True
):
    """
    Generates a file from a template file.
    The template file may contain template variables that will be substituted
    with the provided values in the generated output file.
    In the template file, template variable names are delimited by "@"'s,
    e.g., "@var@".

    :param template_file The template file path.
    :param output_file The generated output file path.
    :param variable_substitutions The mapping from template variable name to value.
    :param strict Whether to require the set of template variable names in the file and the keys of
                  `variable_substitutions` to be equal.
    """
    with open(template_file, mode="r") as template:
        content = template.read()

    variables_in_file = set()

    def replace_template_variable(match):
        variable_name = match.group(1)
        variables_in_file.add(variable_name)
        return variable_substitutions.get(variable_name, match.group(0))

    content = _template_variable_pattern.sub(replace_template_variable, content)

    if strict and variables_in_file != variable_substitutions.keys():
        variables_in_substitutions = set(variable_substitutions.keys())
        raise ValueError(
            f"Template file variables and substitution variables do not match. "
            f"Only in template file: {sorted(variables_in_file - variables_in_substitutions)}. "
            f"Only in substitutions: {sorted(variables_in_substitutions - variables_in_file)}."
        )

    with open(output_file, mode="w") as output:
        output.write(content)


def assemble_pod_package(
    staging_dir: Path, xcframework_dir: Path, public_headers_dir: Path, framework_info_file: Path, pod_version: str
):
    staging_dir = staging_dir.resolve()
    xcframework_dir = xcframework_dir.resolve(strict=True)
    public_headers_dir = public_headers_dir.resolve(strict=True)
    framework_info_file = framework_info_file.resolve(strict=True)

    print(f"Assembling files in staging directory: {staging_dir}")
    if staging_dir.exists():
        print("Warning: staging directory already exists", file=sys.stderr)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # copy files to staging dir
    shutil.copyfile(_repo_dir / "LICENSE", staging_dir / "LICENSE")
    shutil.copytree(xcframework_dir, staging_dir / xcframework_dir.name, dirs_exist_ok=True)
    shutil.copytree(public_headers_dir, staging_dir / public_headers_dir.name, dirs_exist_ok=True)

    # generate podspec
    pod_name = "onnxruntime-extensions-c"
    pod_summary = "ONNX Runtime Extensions C/C++ Pod"
    pod_description = "Pod containing pre and post processing custom ops for onnxruntime."
    with open(framework_info_file, mode="r") as f:
        framework_info = json.load(f)
        ios_deployment_target = framework_info["apple_deployment_target"]

    podspec_variable_substitutions = {
        "NAME": pod_name,
        "VERSION": pod_version,
        "SUMMARY": pod_summary,
        "DESCRIPTION": pod_description,
        "IOS_DEPLOYMENT_TARGET": ios_deployment_target,
        "LICENSE_FILE": "LICENSE",
        "XCFRAMEWORK_DIR": xcframework_dir.name,
        "PUBLIC_HEADERS_DIR": public_headers_dir.name,
    }

    gen_file_from_template(
        _script_dir / "podspec.template", staging_dir / f"{pod_name}.podspec", podspec_variable_substitutions
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assembles the files for the pod package in a staging directory. "
        "This directory can be validated (e.g., with `pod lib lint`) and then zipped to create a package for release.",
    )

    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="Path to the pod output staging directory.",
    )

    parser.add_argument(
        "--pod-version",
        required=True,
        help="The pod's version.",
    )

    input_paths_group = parser.add_argument_group(description="Input path arguments.")

    input_paths_group.add_argument(
        "--xcframework-output-dir",
        type=Path,
        help=f"Path to the output directory produced by {_script_dir / 'build_xcframework.py'}. "
        "Specify either this argument or the other input path arguments.",
    )
    input_paths_group.add_argument(
        "--xcframework-dir",
        type=Path,
        help="Path to the onnxruntime_extensions xcframework to include in the pod.",
    )
    input_paths_group.add_argument(
        "--public-headers-dir",
        type=Path,
        help="Path to the public header directory to include in the pod.",
    )
    input_paths_group.add_argument(
        "--framework-info-file",
        type=Path,
        help="Path to the framework_info.json file containing additional values for the podspec. "
        "This file should be generated by CMake in the build directory.",
    )

    args = parser.parse_args()

    assert bool(args.xcframework_output_dir is not None) ^ bool(
        args.xcframework_dir is not None
        and args.public_headers_dir is not None
        and args.framework_info_file is not None
    ), "Specify either --xcframework-output-dir OR all of --xcframework-dir, --public-headers-dir, and "
    "--framework-info-file."

    if args.xcframework_output_dir is not None:
        args.xcframework_dir = args.xcframework_output_dir / "onnxruntime_extensions.xcframework"
        args.public_headers_dir = args.xcframework_output_dir / "Headers"
        args.framework_info_file = args.xcframework_output_dir / "framework_info.json"

    return args


def main():
    args = parse_args()

    assemble_pod_package(
        staging_dir=args.staging_dir,
        xcframework_dir=args.xcframework_dir,
        public_headers_dir=args.public_headers_dir,
        framework_info_file=args.framework_info_file,
        pod_version=args.pod_version,
    )


if __name__ == "__main__":
    main()
