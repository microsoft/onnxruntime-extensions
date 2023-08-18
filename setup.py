# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

from setuptools import setup, find_packages
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext

import re
import os
import sys
import setuptools
import pathlib
import subprocess

from textwrap import dedent

TOP_DIR = os.path.dirname(__file__) or os.getcwd()
PACKAGE_NAME = 'onnxruntime_extensions'
VSINSTALLDIR_NAME = 'VSINSTALLDIR'


def load_vsdevcmd():
    if os.environ.get(VSINSTALLDIR_NAME) is None:
        stdout, _ = subprocess.Popen([
            'powershell', ' -noprofile', '-executionpolicy',
            'bypass', '-f', TOP_DIR + '/tools/get_vsdevcmd.ps1', '-outputEnv', '1'],
            stdout=subprocess.PIPE, shell=False, universal_newlines=True).communicate()
        for line in stdout.splitlines():
            kv_pair = line.split('=')
            if len(kv_pair) == 2:
                os.environ[kv_pair[0]] = kv_pair[1]
    else:
        import shutil
        if shutil.which('cmake') is None:
            raise SystemExit(
                "Cannot find cmake in the executable path, "
                "please run this script under Developer Command Prompt for VS.")


def read_git_refs():
    release_branch = False
    stdout, _ = subprocess.Popen(
        ['git'] + ['log', '-1', '--format=%H'],
        cwd=TOP_DIR,
        stdout=subprocess.PIPE, universal_newlines=True).communicate()
    HEAD = dedent(stdout.splitlines()[0]).strip('\n\r')
    stdout, _ = subprocess.Popen(
        ['git'] + ['show-ref', '--head'],
        cwd=TOP_DIR,
        stdout=subprocess.PIPE, universal_newlines=True).communicate()
    for _ln in stdout.splitlines():
        _ln = dedent(_ln).strip('\n\r')
        if _ln.startswith(HEAD):
            _, _2 = _ln.split(' ')
            if _2.startswith('refs/remotes/origin/rel-'):
                release_branch = True
    return release_branch, HEAD


class BuildCMakeExt(_build_ext):

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            if extension.name == 'onnxruntime_extensions._extensions_pydll':
                self.build_cmake(extension)

    def build_cmake(self, extension):
        project_dir = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(extension.name)).absolute()

        config = 'RelWithDebInfo' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(ext_fullpath.parent.absolute()),
            '-DOCOS_BUILD_PYTHON=ON',
            '-DOCOS_PYTHON_MODULE_PATH=' + str(ext_fullpath),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        if os.environ.get('OCOS_NO_OPENCV') == '1':
            # Disabling openCV can drastically reduce the build time.
            cmake_args += [
                '-DOCOS_ENABLE_OPENCV_CODECS=OFF',
                '-DOCOS_ENABLE_CV2=OFF',
                '-DOCOS_ENABLE_VISION=OFF']

        # explicitly set the flag for AzureOp, despite the default value in CMakeLists.txt
        azure_flag = "ON" if os.environ.get('OCOS_ENABLE_AZURE') == '1' else None
        if azure_flag is None:
            # OCOS_NO_AZURE will be ignored if OCOS_ENABLE_AZURE is set.
            azure_flag = "OFF" if os.environ.get('OCOS_NO_AZURE') == '1' else None
        if azure_flag is not None:
            cmake_args += ['-DOCOS_ENABLE_AZURE=' + azure_flag]
            print("=> AzureOp build flag: " + azure_flag)

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if sys.platform != "win32":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # overwrite the Python module info if the auto-detection doesn't work.
        # export Python3_INCLUDE_DIRS=/opt/python/cp38-cp38
        # export Python3_LIBRARIES=/opt/python/cp38-cp38
        for env in ['Python3_INCLUDE_DIRS', 'Python3_LIBRARIES']:
            if env in os.environ:
                cmake_args.append("-D%s=%s" % (env, os.environ[env]))

        if self.debug:
            cmake_args += ['-DCC_OPTIMIZE=OFF']

        # the parallel build has to be limited on some Linux VM machine.
        cpu_number = os.environ.get('CPU_NUMBER')
        build_args = [
            '--config', config,
            '--parallel' + ('' if cpu_number is None else ' ' + cpu_number)
        ]
        cmake_exe = 'cmake'
        # unlike Linux/macOS, cmake pip package on Windows fails to build some 3rd party dependencies.
        # so we have to use the cmake installed from Visual Studio.
        if os.environ.get(VSINSTALLDIR_NAME):
            cmake_exe = os.environ[VSINSTALLDIR_NAME] + \
                        'Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe'
            # Add this cmake directory into PATH to make sure the child-process still find it.
            os.environ['PATH'] = os.path.dirname(cmake_exe) + os.pathsep + os.environ['PATH']

        self.spawn([cmake_exe, '-S', str(project_dir), '-B', str(build_temp)] + cmake_args)
        if not self.dry_run:
            self.spawn([cmake_exe, '--build', str(build_temp)] + build_args)


class Build(_build):
    def initialize_options(self) -> None:
        super().initialize_options()
        if os.environ.get('OCOS_SCB_DEBUG', None) == '1':
            self.debug = True

    def finalize_options(self) -> None:
        # There is a bug in setuptools that prevents the build get the right platform name from arguments.
        # So, it cannot generate the correct wheel with the right arch in Official release pipeline.
        # Force plat_name to be 'win-amd64' in Windows to fix that.
        # Since extensions cmake is only available on x64 for Windows now, it is not a problem to hardcode it.
        if sys.platform == "win32" and "arm" not in sys.version.lower():
            self.plat_name = "win-amd64"
        super().finalize_options()


def read_requirements():
    with open(os.path.join(TOP_DIR, "requirements.txt"), "r", encoding="utf-8") as f:
        requirements = [_ for _ in [dedent(_) for _ in f.readlines()] if _ is not None]
    return requirements


# read version from the package file.
def read_version():
    version_str = '1.0.0'
    with (open(os.path.join(TOP_DIR, 'version.txt'), "r")) as f:
        version_str = f.readline().strip()

    # special handling for Onebranch building
    if os.getenv('BUILD_SOURCEBRANCHNAME', "").startswith('rel-'):
        return version_str

    # is it a dev build or release?
    rel_br, cid = read_git_refs() if os.path.isdir(
        os.path.join(TOP_DIR, '.git')) else (True, None)

    if rel_br:
        return version_str

    build_id = os.getenv('BUILD_BUILDID', None)
    if build_id is not None:
        version_str += '.{}'.format(build_id)
    else:
        version_str += '+' + cid[:7]
    return version_str


def write_py_version(ortx_version):
    text = ["# Generated by setup.py, DON'T MANUALLY UPDATE IT!\n",
            "__version__ = \"{}\"\n".format(ortx_version)]
    with (open(os.path.join(TOP_DIR, 'onnxruntime_extensions/_version.py'), "w")) as _f:
        _f.writelines(text)


if sys.platform == "win32":
    load_vsdevcmd()

ext_modules = [
    setuptools.extension.Extension(
        name=str('onnxruntime_extensions._extensions_pydll'),
        sources=[])
]

packages = find_packages()
package_dir = {k: os.path.join('.', k.replace(".", "/")) for k in packages}
package_data = {
    "onnxruntime_extensions": ["*.so", "*.pyd"],
}

long_description = ''
with open(os.path.join(TOP_DIR, "README.md"), 'r', encoding="utf-8") as _f:
    long_description += _f.read()
    start_pos = long_description.find('# Introduction')
    start_pos = 0 if start_pos < 0 else start_pos
    end_pos = long_description.find('# Contributing')
    long_description = long_description[start_pos:end_pos]
ortx_version = read_version()
write_py_version(ortx_version)

setup(
    name=PACKAGE_NAME,
    version=ortx_version,
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    description="ONNXRuntime Extensions",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT License',
    author='Microsoft Corporation',
    author_email='onnxruntime@microsoft.com',
    url='https://github.com/microsoft/onnxruntime-extensions',
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=BuildCMakeExt, build=Build),
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
        "Programming Language :: Python :: Implementation :: CPython",
        'License :: OSI Approved :: MIT License'
    ]
)
