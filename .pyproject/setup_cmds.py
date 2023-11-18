# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import re
import os
import sys
import shlex
import pathlib

from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop

VSINSTALLDIR_NAME = 'VSINSTALLDIR'


class CommandMixin:
    user_options = [
        # ('use-cuda', None, 'enable CUDA kernels on building extensions'),
        # ('no-azure', None, 'disable AzureOp kernels on building extensions'),
        # ('no-opencv', None, 'disable OpenCV based kernels on building extensions')
        ('ortx-user-option=', None, "extensions options for kernel building")
    ]

    supported_cmds = ['build', 'develop', 'install']

    config_settings = None

    @staticmethod
    def assign_options(target, distr):
        active_obj = None
        if len(distr.commands) > 0:
            active_obj = distr.get_command_obj(distr.commands[0], create=0)

        if active_obj is not None and isinstance(active_obj, CommandMixin):
            for _opt in CommandMixin.user_options:
                opt_name = _opt[0].replace('-', '_')
                value = getattr(active_obj, opt_name)
                setattr(target, opt_name, value)

    @staticmethod
    def environ_to_options():
        build_opt = os.environ.get('OCOS_SETUP_OPTIONS', None)
        if build_opt is not None:
            opt_list = shlex.split(build_opt)
            cmd_found = False
            for _cmd in CommandMixin.supported_cmds:
                if _cmd in sys.argv[1:]:
                    cmd_found = True
                    break
            if cmd_found:
                print("=> Setup options from environment variable: " + build_opt)
                print("=> Setup options from sys.argv: " + str(sys.argv))
                sys.argv += opt_list

    def __init__(self, dist):
        self.ortx_user_option = None
        super().__init__(dist)

    def initialize_options(self) -> None:
        super().initialize_options()

    def finalize_options(self) -> None:
        if self.ortx_user_option is not None:
            if CommandMixin.config_settings is None:
                CommandMixin.config_settings = {"ortx-user-option": self.ortx_user_option}
            else:
                raise RuntimeError("Cannot pass ortx-user-option both as the command args and in backend API.")
            if self is self.get_finalized_command('build', create=1):
                # There is a bug in setuptools that prevents the build get the right platform name from arguments.
                # So, it cannot generate the correct wheel with the right arch in Official release pipeline.
                # Force plat_name to be 'win-amd64' in Windows to fix that,
                # since extensions cmake is only available on x64 for Windows now, it is not a problem to hardcode it.
                if sys.platform == "win32" and "arm" not in sys.version.lower():
                    self.plat_name = "win-amd64"
                if os.environ.get('OCOS_SCB_DEBUG', None) == '1':
                    self.debug = True

        super().finalize_options()
    #
    # def finalize_options(self) -> None:
    #     if isinstance(self, _build):
    #         # There is a bug in setuptools that prevents the build get the right platform name from arguments.
    #         # So, it cannot generate the correct wheel with the right arch in Official release pipeline.
    #         # Force plat_name to be 'win-amd64' in Windows to fix that,
    #         # since extensions cmake is only available on x64 for Windows now, it is not a problem to hardcode it.
    #         if sys.platform == "win32" and "arm" not in sys.version.lower():
    #             self.plat_name = "win-amd64"
    #         if os.environ.get('OCOS_SCB_DEBUG', None) == '1':
    #             self.debug = True
    #
    #     super().finalize_options()

    # cmd_objs = self.distribution.command_obj
    # # bridge the options from develop to build
    # if "develop" in self.distribution.commands or \
    #         'install' in self.distribution.commands:
    #     build_cmd = cmd_objs.get('build', None)
    #     if build_cmd:
    #         build_cmd.no_cuda = self.no_cuda
    #         build_cmd.no_azure = self.no_azure
    #         build_cmd.no_opencv = self.no_opencv

    # def run(self):
    #     build_ext_cmd = None
    #     try:
    #         cmd_objs = self.distribution.command_obj
    #         build_ext_cmd = cmd_objs.get('build_ext', None)
    #     except AttributeError:
    #         pass
    #
    #     if build_ext_cmd:
    #         setattr(build_ext_cmd, 'no_cuda', self.no_cuda)
    #         setattr(build_ext_cmd, 'no_azure', self.no_azure)
    #         setattr(build_ext_cmd, 'no_opencv', self.no_opencv)
    #
    #     return super().run()


class CmdDevelop(CommandMixin, _develop):
    user_options = getattr(_develop, 'user_options', []) + CommandMixin.user_options


# class CmdBuild(CommandMixin, _build):
#     user_options = getattr(_build, 'user_options', []) + CommandMixin.user_options
#
#     def initialize_options(self) -> None:
#         build_opt = os.environ.get('OCOS_SETUP_OPTIONS', None)
#         if build_opt is not None:
#             opt_list = shlex.split(build_opt)
#             cmd_found = True
#             # for _cmd in CommandMixin.supported_cmds:
#             #     if _cmd in sys.argv[1:]:
#             #         cmd_found = True
#             #         break
#             if cmd_found:
#                 print("=> Setup options from environment variable: " + build_opt)
#                 print("=> Setup options from sys.argv: " + str(sys.argv))
#                 sys.argv += opt_list
#
#         super(_build, self).initialize_options()
#
#     # noinspection PyAttributeOutsideInit
#     def finalize_options(self) -> None:
#         if os.environ.get('OCOS_SCB_DEBUG', None) == '1':
#             self.debug = 1
#         # There is a bug in setuptools that prevents the build get the right platform name from arguments.
#         # So, it cannot generate the correct wheel with the right arch in Official release pipeline.
#         # Force plat_name to be 'win-amd64' in Windows to fix that,
#         # since extensions cmake is only available on x64 for Windows now, it is not a problem to hardcode it.
#         if sys.platform == "win32" and "arm" not in sys.version.lower():
#             self.plat_name = "win-amd64"
#
#         super(_build, self).finalize_options()


class CmdBuildCMakeExt(_build_ext):

    # noinspection PyAttributeOutsideInit
    def initialize_options(self):
        super().initialize_options()
        self.use_cuda = None
        self.no_azure = None
        self.no_opencv = None

    def _parse_options(self, options):
        for segment in options.split(','):
            key = segment
            if '=' in segment:
                key, value = segment.split('=')
            else:
                value = 1

            key = key.replace('-', '_')
            setattr(self, key, value)
        return self

    # noinspection PyAttributeOutsideInit
    def finalize_options(self) -> None:
        if CommandMixin.config_settings is not None:
            self._parse_options(CommandMixin.config_settings.get("ortx-user-option", ""))

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
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(extension.name)).absolute()

        config = 'RelWithDebInfo' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(ext_fullpath.parent.absolute()),
            '-DOCOS_BUILD_PYTHON=ON',
            '-DOCOS_PYTHON_MODULE_PATH=' + str(ext_fullpath),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

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


ortx_cmdclass = dict(develop=CmdDevelop,
                     build_ext=CmdBuildCMakeExt)
