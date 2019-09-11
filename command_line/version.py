from __future__ import absolute_import, division, print_function

import os

import dials
from dials.util.version import dials_version


def version():
    print(dials_version())
    print("Installed in: %s" % os.path.split(dials.__file__)[0])


version()
