# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import re
import os
import sys
import pathlib
import subprocess

from textwrap import dedent
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop

VSINSTALLDIR_NAME = 'VSINSTALLDIR'
ORTX_USER_OPTION = 'ortx-user-option'


def _load_cuda_version():
    nvcc_path = 'nvcc'
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path is not None:
        nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc')
    try:
        output = subprocess.check_output([nvcc_path, "--version"], stderr=subprocess.STDOUT).decode("utf-8")
        pattern = r"\bV(\d+\.\d+\.\d+)\b"
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, OSError):
        pass

    return None


def _load_nvidia_smi():
    try:
        outputs = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT).decode("utf-8").splitlines()
        output = outputs[0] if outputs else ""
        arch = output.strip().replace('.', '')
        return arch if arch.isdigit() else None
    except (subprocess.CalledProcessError, OSError):
        pass

    return None


def _load_vsdevcmd(project_root):
    if os.environ.get(VSINSTALLDIR_NAME) is None:
        stdout, _ = subprocess.Popen([
            'powershell', ' -noprofile', '-executionpolicy',
            'bypass', '-f', project_root + '/tools/get_vsdevcmd.ps1', '-outputEnv', '1'],
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


def prepare_env(project_root):
    if sys.platform == "win32":
        _load_vsdevcmd(project_root)


def read_git_refs(project_root):
    release_branch = False
    stdout, _ = subprocess.Popen(
        ['git'] + ['log', '-1', '--format=%H'],
        cwd=project_root,
        stdout=subprocess.PIPE, universal_newlines=True).communicate()
    HEAD = dedent(stdout.splitlines()[0]).strip('\n\r')
    stdout, _ = subprocess.Popen(
        ['git'] + ['show-ref', '--head'],
        cwd=project_root,
        stdout=subprocess.PIPE, universal_newlines=True).communicate()
    for _ln in stdout.splitlines():
        _ln = dedent(_ln).strip('\n\r')
        if _ln.startswith(HEAD):
            _, _2 = _ln.split(' ')
            if _2.startswith('refs/remotes/origin/rel-'):
                release_branch = True
    return release_branch, HEAD


class CommandMixin:
    user_options = [
        (ORTX_USER_OPTION + '=', None, "extensions options for kernel building")
    ]
    config_settings = None

    # noinspection PyAttributeOutsideInit
    def initialize_options(self) -> None:
        super().initialize_options()
        self.ortx_user_option = None

    def finalize_options(self) -> None:
        if self.ortx_user_option is not None:
            if CommandMixin.config_settings is None:
                CommandMixin.config_settings = {
                    ORTX_USER_OPTION: self.ortx_user_option}
            else:
                raise RuntimeError(
                    f"Cannot pass {ORTX_USER_OPTION} several times, like as the command args and in backend API.")

        super().finalize_options()


class CmdDevelop(CommandMixin, _develop):
    user_options = getattr(_develop, 'user_options', []
                           ) + CommandMixin.user_options


class CmdBuild(CommandMixin, _build):
    user_options = getattr(_build, 'user_options', []) + \
                   CommandMixin.user_options

    # noinspection PyAttributeOutsideInit
    def finalize_options(self) -> None:
        # There is a bug in setuptools that prevents the build get the right platform name from arguments.
        # So, it cannot generate the correct wheel with the right arch in Official release pipeline.
        # Force plat_name to be 'win-amd64' in Windows to fix that,
        # since extensions cmake is only available on x64 for Windows now, it is not a problem to hardcode it.
        if sys.platform == "win32" and "arm" not in sys.version.lower():
            self.plat_name = "win-amd64"
        if os.environ.get('OCOS_SCB_DEBUG', None) == '1':
            self.debug = True
        super().finalize_options()


class CmdBuildCMakeExt(_build_ext):

    # noinspection PyAttributeOutsideInit
    def initialize_options(self):
        super().initialize_options()
        self.use_cuda = None
        self.no_azure = None
        self.no_opencv = None
        self.cc_debug = None
        self.cuda_archs = None
        self.ort_pkg_dir = None

    def _parse_options(self, options):
        for segment in options.split(','):
            if not segment:
                continue
            key = segment
            if '=' in segment:
                key, value = segment.split('=')
            else:
                value = 1

            key = key.replace('-', '_')
            if not hasattr(self, key):
                raise RuntimeError(
                    f"Unknown {ORTX_USER_OPTION} option value: {key}")
            setattr(self, key, value)
        return self

    def finalize_options(self) -> None:
        if CommandMixin.config_settings is not None:
            self._parse_options(
                CommandMixin.config_settings.get(ORTX_USER_OPTION, ""))
            if self.cc_debug:
                self.debug = True
        super().finalize_options()

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
        ext_fullpath = pathlib.Path(
            self.get_ext_fullpath(extension.name)).absolute()

        config = 'RelWithDebInfo' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            str(ext_fullpath.parent.absolute()),
            '-DOCOS_ENABLE_CTEST=OFF',
            '-DOCOS_BUILD_PYTHON=ON',
            '-DOCOS_PYTHON_MODULE_PATH=' + str(ext_fullpath),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        if self.ort_pkg_dir:
            cmake_args += ['-DONNXRUNTIME_PKG_DIR=' + self.ort_pkg_dir]

        if self.no_opencv:
            # Disabling openCV can drastically reduce the build time.
            cmake_args += [
                '-DOCOS_ENABLE_OPENCV_CODECS=OFF',
                '-DOCOS_ENABLE_CV2=OFF',
                '-DOCOS_ENABLE_VISION=OFF']

        if self.no_azure is not None:
            azure_flag = "OFF" if self.no_azure == 1 else "ON"
            cmake_args += ['-DOCOS_ENABLE_AZURE=' + azure_flag]
            print("=> AzureOp build flag: " + azure_flag)

        if self.use_cuda is not None:
            cuda_flag = "OFF" if self.use_cuda == 0 else "ON"
            cmake_args += ['-DOCOS_USE_CUDA=' + cuda_flag]
            print("=> CUDA build flag: " + cuda_flag)
            if cuda_flag == "ON":
                cuda_ver = _load_cuda_version()
                if cuda_ver is None:
                    raise RuntimeError("Cannot find nvcc in your env:path, use-cuda doesn't work")
                if sys.platform == "win32":
                    cuda_path = os.environ.get("CUDA_PATH")
                    cmake_args += [f'-T cuda={cuda_path}']
                    # TODO: temporarily add a flag for MSVC 19.40
                    cmake_args += ['-DCMAKE_CUDA_FLAGS_INIT=-allow-unsupported-compiler']
                f_ver = ext_fullpath.parent / "_version.py"
                with f_ver.open('a') as _f:
                    _f.writelines(["\n", f"cuda = \"{cuda_ver}\"", "\n"])

                if self.cuda_archs is not None:
                    cmake_args += ['-DCMAKE_CUDA_ARCHITECTURES=' + self.cuda_archs]
                else:
                    smi = _load_nvidia_smi()
                    if not smi:
                        raise RuntimeError(f"Cannot detect the CUDA archs from your machine, please specify it by yourself.")
                    cmake_args += ['-DCMAKE_CUDA_ARCHITECTURES=' + smi]

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if sys.platform != "win32":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithread automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = os.path.join(
                        ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        if sys.platform.startswith("darwin"):
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15"]
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]


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
            os.environ['PATH'] = os.path.dirname(
                cmake_exe) + os.pathsep + os.environ['PATH']

        self.spawn([cmake_exe, '-S', str(project_dir),
                    '-B', str(build_temp)] + cmake_args)
        if not self.dry_run:
            self.spawn([cmake_exe, '--build', str(build_temp)] + build_args)


ortx_cmdclass = dict(build=CmdBuild,
                     develop=CmdDevelop,
                     build_ext=CmdBuildCMakeExt)
