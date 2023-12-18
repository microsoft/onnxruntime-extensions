# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import os
import sys
from setuptools import build_meta as _orig
from setuptools.build_meta import *  # noqa: F401, F403

# add the current directory to the path, so we can import setup_cmds.py
sys.path.append(os.path.dirname(__file__))
import cmdclass as _cmds  # noqa: E402


def build_wheel(wheel_directory, config_settings=None,
                metadata_directory=None):
    _cmds.CommandMixin.config_settings = config_settings

    return _orig.build_wheel(
        wheel_directory, config_settings,
        metadata_directory
    )


def build_editable(wheel_directory, config_settings=None,
                   metadata_directory=None):
    _cmds.CommandMixin.config_settings = config_settings

    return _orig.build_editable(
        wheel_directory, config_settings,
        metadata_directory
    )
