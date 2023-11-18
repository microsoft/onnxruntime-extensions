import os
import sys
from setuptools import build_meta as _orig
from setuptools.build_meta import *  # noqa: F403

sys.path.append(os.path.dirname(__file__))
# add the current directory to the path, so we can import setup_cmds.py
import setup_cmds as _cmds  # noqa: E402


#
# def _setup_option_config(config_settings) -> List[str]:
#     cfg = config_settings or {}
#     opts = cfg.get('ocos-setup-options') or []
#     return shlex.split(opts) if isinstance(opts, str) else opts


def build_wheel(wheel_directory, config_settings=None,
                metadata_directory=None):
    _cmds.CommandMixin.config_settings = config_settings
    # opt = _setup_option_config(config_settings)
    # if opt:
    #     os.environ['OCOS_SETUP_OPTIONS'] = shlex.join(opt)

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
