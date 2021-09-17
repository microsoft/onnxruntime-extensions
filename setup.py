# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

import os
import sys
import setuptools
import pathlib
import subprocess

from textwrap import dedent


TOP_DIR = os.path.dirname(__file__) or os.getcwd()
PACKAGE_NAME = 'onnxruntime_extensions'


def load_msvcvar():
    if os.environ.get('vcvars'):
        stdout, _ = subprocess.Popen([
            'cmd', '/q', '/c', '(%vcvars% & set)'],
            stdout=subprocess.PIPE, shell=True, universal_newlines=True).communicate()
        for line in stdout.splitlines():
            kv_pair = line.split('=')
            if len(kv_pair) == 2:
                os.environ[kv_pair[0]] = kv_pair[1]
    else:
        import shutil
        if shutil.which('cmake') is None:
            raise SystemExit(
                "Cannot find cmake in the executable path, " +
                "please install one or specify the environement variable VCVARS to the path of VS vcvars64.bat.")


def read_git_refs(git_args):
    stdout, _ = subprocess.Popen(
            ['git'] + git_args,
            cwd=TOP_DIR,
            stdout=subprocess.PIPE, universal_newlines=True).communicate()
    for _ln in stdout.splitlines():
        _ln = dedent(_ln).strip('\n\r')
        if _ln:
            return _ln
    return ''


class BuildCMakeExt(_build_ext):

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            if extension.name == 'onnxruntime_extensions._ortcustomops':
                self.build_cmake(extension)

    def build_cmake(self, extension):
        project_dir = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(extension.name))

        config = 'RelWithDebInfo' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(ext_fullpath.parent.absolute()),
            '-DOCOS_ENABLE_PYTHON=ON',
            '-DOCOS_ENABLE_CTEST=OFF',
            '-DOCOS_EXTENTION_NAME=' + pathlib.Path(self.get_ext_filename(extension.name)).name,
            '-DCMAKE_BUILD_TYPE=' + config
        ]
        # Uses to overwrite 
        # export Python3_INCLUDE_DIRS=/opt/python/cp38-cp38
        # export Python3_LIBRARIES=/opt/python/cp38-cp38
        for env in ['Python3_INCLUDE_DIRS', 'Python3_LIBRARIES']:
            if env in os.environ:
                cmake_args.append("-D%s=%s" % (env, os.environ[env]))

        if self.debug:
            cmake_args += ['-DCC_OPTIMIZE=OFF']

        build_args = [
            '--config', config,
            '--parallel'
        ]

        self.spawn(['cmake', '-S', str(project_dir), '-B', str(build_temp)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', str(build_temp)] + build_args)

        if sys.platform == "win32":
            self.copy_file(build_temp / config / 'ortcustomops.dll',
                           self.get_ext_filename(extension.name))


def read_requirements():
    with open(os.path.join(TOP_DIR, "requirements.txt"), "r") as f:
        requirements = [_ for _ in [dedent(_) for _ in f.readlines()] if _ is not None]
    return requirements


# read version from the package file.
def read_version():
    version_str = '1.0.0'
    with (open(os.path.join(TOP_DIR, 'onnxruntime_extensions/_version.py'), "r")) as f:
        line = [_ for _ in [dedent(_) for _ in f.readlines()] if _.startswith("__version__")]
        if len(line) > 0:
            version_str = line[0].split('=')[1].strip('" \n\r')

    # is it a nightly or dev build?
    if os.path.isdir('.git') and \
       not read_git_refs(['rev-parse', '--abbrev-ref', 'HEAD']).startswith('rel-'):
        # append a git commit id from git remote repo, while the local change ids are skipped.
        version_str += '+' + read_git_refs(['rev-parse', 'HEAD'])[:7]
    return version_str


if sys.platform == "win32":
    load_msvcvar()


ext_modules = [
    setuptools.extension.Extension(
        name=str('onnxruntime_extensions._ortcustomops'),
        sources=[])
]

packages = find_packages()
package_dir = {k: os.path.join('.', k.replace(".", "/")) for k in packages}
package_data = {
    "onnxruntime_extensions": ["*.dll", "*.so", "*.pyd"],
}

long_description = ''
with open(os.path.join(TOP_DIR, "README.md"), 'r') as f:
    long_description = f.read()
    start_pos = long_description.find('# Introduction')
    start_pos = 0 if start_pos < 0 else start_pos
    end_pos = long_description.find('# Contributing')
    long_description = long_description[start_pos:end_pos]

setup(
    name=PACKAGE_NAME,
    version=read_version(),
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    description="ONNXRuntime Extensions",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT License',
    author='Microsoft Corporation',
    author_email='onnx@microsoft.com',
    url='https://github.com/microsoft/onnxruntime-extensions',
    ext_modules=ext_modules,
    cmdclass=dict(
        build_ext=BuildCMakeExt,
        ),
    include_package_data=True,
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        "Programming Language :: C++",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: Implementation :: CPython",
        'License :: OSI Approved :: MIT License'
    ],
)
