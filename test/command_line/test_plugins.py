from __future__ import absolute_import, division, print_function

import dials.command_line.plugins


def test_plugin_setup_is_valid():
    assert dials.command_line.plugins.installation_is_valid()
