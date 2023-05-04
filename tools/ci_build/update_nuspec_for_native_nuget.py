# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import xml.etree.ElementTree as ElementTree
import argparse
import sys
from pathlib import Path


def update_nuspec(args):
    print(f"Updating {args.nuspec_path}")

    # preserve comments
    tree = ElementTree.parse(args.nuspec_path,
                             parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(insert_comments=True)))
    root = tree.getroot()

    # update version and commit id
    packages_node = root.findall('metadata')[0]
    for package_item in packages_node:
        if package_item.tag == "version" and args.package_version:
            if args.is_release_build:
                package_item.text = args.package_version
            else:
                import datetime
                now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                package_item.text = f"{args.package_version}-dev-{now}-{args.commit_id}"
        elif package_item.tag == "repository" and args.commit_id:
            package_item.attrib['commit'] = args.commit_id

    # format of indent
    py_version = sys.version_info
    if py_version > (3, 9):
        ElementTree.indent(root)

    tree.write(args.nuspec_path, encoding='utf-8', xml_declaration=True)
    return


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime extensions create nuget spec script (for hosting native shared library artifacts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_nuspec = Path(__file__).resolve().parents[2] / "nuget" / "NativeNuget.nuspec"

    # Main arguments
    parser.add_argument("--package_version", default='', help="ORT extensions package version. e.g.: 1.0.0")
    parser.add_argument("--nuspec_path", type=Path, default=default_nuspec,
                        help="Path to nuspec file to update.")
    parser.add_argument("--commit_id", required=True, help="The last commit id included in this package.")
    parser.add_argument("--is_release_build", default="False", type=str, help="If it's a release build.")

    args = parser.parse_args()
    args.nuspec_path = args.nuspec_path.resolve(strict=True)
    args.is_release_build = args.is_release_build.lower() == "true"
    print("used args:", args)

    return args


def main():
    args = parse_arguments()
    update_nuspec(args)


if __name__ == "__main__":
    sys.exit(main())
