# -*- coding: utf-8 -*-

###########################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop
from setuptools.command.build_py import build_py as _build_py
from distutils.core import setup
from contextlib import contextmanager

import os
import sys
import setuptools
import pathlib
import subprocess


TOP_DIR = os.path.dirname(__file__)


@contextmanager
def chdir(path):
    orig_path = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(orig_path)


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


class BuildCMakeExt(_build_ext):

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            if extension.name == 'onnxruntime_customops._ortcustomops':
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
            '-DOCOS_EXTENTION_NAME=' + pathlib.Path(self.get_ext_filename(extension.name)).name,
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        if self.debug:
            cmake_args += ['-DCC_OPTIMIZE=OFF']

        build_args = [
            '--config', config,
            '--parallel'
        ]

        with chdir(build_temp):
            self.spawn(['cmake', str(project_dir)] + cmake_args)
            if not self.dry_run:
                self.spawn(['cmake', '--build', '.'] + build_args)

        if sys.platform == "win32":
            self.copy_file(build_temp / config / 'ortcustomops.dll',
                           self.get_ext_filename(extension.name))


class BuildPy(_build_py):
    def run(self):
        super().run()


class BuildDevelop(_develop):
    def run(self):
        super().run()


def read_requirements():
    with open(os.path.join(TOP_DIR, "requirements.txt"), "r") as f:
        requirements = [_ for _ in [_.strip("\r\n ")
                                    for _ in f.readlines()] if _ is not None]
    return requirements


# read version from the package file.
def read_version():
    version_str = '1.0.0'
    with (open(os.path.join(TOP_DIR, 'onnxruntime_customops/__init__.py'), "r")) as f:
        line = [_ for _ in [_.strip("\r\n ")
                            for _ in f.readlines()] if _.startswith("__version__")]
        if len(line) > 0:
            version_str = line[0].split('=')[1].strip('" ')
    return version_str


if sys.platform == "win32":
    load_msvcvar()


ext_modules = [
    setuptools.extension.Extension(
        name=str('onnxruntime_customops._ortcustomops'),
        sources=[])
]


setup(
    name='onnxruntime_customops',
    version=read_version(),
    packages=['onnxruntime_customops'],
    description="ONNXRuntime Custom Operator Library",
    long_description=open(os.path.join(os.getcwd(), "README.md"), 'r').read(),
    long_description_content_type='text/markdown',
    license='MIT License',
    author='Microsoft Corporation',
    author_email='onnx@microsoft.com',
    url='https://github.com/microsoft/ortcustomops',
    ext_modules=ext_modules,
    cmdclass=dict(
        build_ext=BuildCMakeExt,
        build_py=BuildPy,
        develop=BuildDevelop
        ),
    include_package_data=True,
    install_requires=read_requirements(),
    classifiers=[
        'BuildDevelopment Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        "Programming Language :: C++",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Programming Language :: Python :: Implementation :: CPython",
        'License :: OSI Approved :: MIT License'
    ],
)
