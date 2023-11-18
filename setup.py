# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

from setuptools import setup, find_packages
import os
import sys
import setuptools
import pathlib
import subprocess

from textwrap import dedent

TOP_DIR = os.path.dirname(__file__) or os.getcwd()
PACKAGE_NAME = 'onnxruntime_extensions'

cmds_dir = pathlib.Path(TOP_DIR) / '.pyproject'
sys.path.append(str(cmds_dir))
import setup_cmds as _cmds  # noqa: E402


def load_vsdevcmd():
    if os.environ.get(_cmds.VSINSTALLDIR_NAME) is None:
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


# class CommandMixin:
#     user_options = [
#         ('use-cuda', None, 'enable CUDA kernels on building extensions'),
#         ('no-azure', None, 'disable AzureOp kernels on building extensions'),
#         ('no-opencv', None, 'disable OpenCV based kernels on building extensions')
#     ]

#     supported_cmds = ['build', 'develop', 'install', 'editable_wheel']

#     @staticmethod
#     def assign_options(target, distr):
#         active_obj = None
#         if len(distr.commands) > 0:
#             active_obj = distr.get_command_obj(distr.commands[0], create=0)

#         if active_obj is not None and isinstance(active_obj, CommandMixin):
#             for _opt in CommandMixin.user_options:
#                 opt_name = _opt[0].replace('-', '_')
#                 value = getattr(active_obj, opt_name)
#                 setattr(target, opt_name, value)

#     @staticmethod
#     def environ_to_options():
#         build_opt = os.environ.get('OCOS_SETUP_OPTIONS', None)
#         if  build_opt is not None:
#             opt_list = shlex.split(build_opt)
#             cmd_found = False
#             for _cmd in CommandMixin.supported_cmds:
#                 if _cmd in sys.argv[1:]:
#                     cmd_found = True
#                     break
#             if cmd_found:
#                 print("=> Setup options from environment variable: " + build_opt)
#                 print("=> Setup options from sys.argv: " + str(sys.argv))
#                 sys.argv += opt_list


#     def __init__(self, dist):
#         super().__init__(dist)
#         self.use_cuda = None
#         self.no_azure = None
#         self.no_opencv = None

#     # def initialize_options(self) -> None:
#     #     super().initialize_options()
#     #
#     # def finalize_options(self) -> None:
#     #     if isinstance(self, _build):
#     #         # There is a bug in setuptools that prevents the build get the right platform name from arguments.
#     #         # So, it cannot generate the correct wheel with the right arch in Official release pipeline.
#     #         # Force plat_name to be 'win-amd64' in Windows to fix that,
#     #         # since extensions cmake is only available on x64 for Windows now, it is not a problem to hardcode it.
#     #         if sys.platform == "win32" and "arm" not in sys.version.lower():
#     #             self.plat_name = "win-amd64"
#     #         if os.environ.get('OCOS_SCB_DEBUG', None) == '1':
#     #             self.debug = True
#     #
#     #     super().finalize_options()

#     # cmd_objs = self.distribution.command_obj
#     # # bridge the options from develop to build
#     # if "develop" in self.distribution.commands or \
#     #         'install' in self.distribution.commands:
#     #     build_cmd = cmd_objs.get('build', None)
#     #     if build_cmd:
#     #         build_cmd.no_cuda = self.no_cuda
#     #         build_cmd.no_azure = self.no_azure
#     #         build_cmd.no_opencv = self.no_opencv

#     # def run(self):
#     #     build_ext_cmd = None
#     #     try:
#     #         cmd_objs = self.distribution.command_obj
#     #         build_ext_cmd = cmd_objs.get('build_ext', None)
#     #     except AttributeError:
#     #         pass
#     #
#     #     if build_ext_cmd:
#     #         setattr(build_ext_cmd, 'no_cuda', self.no_cuda)
#     #         setattr(build_ext_cmd, 'no_azure', self.no_azure)
#     #         setattr(build_ext_cmd, 'no_opencv', self.no_opencv)
#     #
#     #     return super().run()

# # update the options from environment variables passed by config_settings
# # CommandMixin.environ_to_options()


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
    cmdclass=_cmds.ortx_cmdclass,
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
