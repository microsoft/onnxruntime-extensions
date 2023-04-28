# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import xml.etree.ElementTree as ElementTree
import argparse
import os
import re
import sys
from pathlib import Path


def update_nuspec_from_local_inplace(args):
    template_nuspec_path = args.sources_path/"nuget"/"NativeNuget.nuspec"
    output_nuspec_path = template_nuspec_path
    tree = ElementTree.parse(template_nuspec_path)
    root = tree.getroot()

    # update version and commit id
    packages_node = root.findall('metadata')[0]
    for package_item in list(packages_node):
        if package_item.tag == "version" and args.package_version:
            package_item.text = args.package_version
        elif package_item.tag == "repository" and args.commit_id:
            package_item.attrib['commit'] = args.commit_id

    # format of indent
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
    parser.add_argument("--sources_path", required=True, type=Path, help="the onnxruntime-extensions repo path.")
    parser.add_argument("--commit_id", default='', help="The last commit id included in this package.")
    parser.add_argument(
        "--is_release_build",
        default='false',
        type=str,
        help="Flag indicating if the build is a release build. Accepted values: true/false.",
    )
    args = parser.parse_args()
    args.sources_path = args.sources_path.resolve()
    return args

def main():
    # Parse arguments
    args = parse_arguments()

    if args.is_release_build.lower() != "true" and args.is_release_build.lower() != "false":
        raise Exception("Only valid options for IsReleaseBuild are: true and false")

    update_nuspec_from_local_inplace(args)


if __name__ == "__main__":
    sys.exit(main())
