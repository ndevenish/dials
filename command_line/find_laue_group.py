#!/usr/bin/env python
# coding: utf-8

"""
Find the most appropriate laue group from symmetry-related integrated data.


"""

from __future__ import absolute_import, print_function, division

import sys
import logging

from dials.util import halraiser

logger = logging.getLogger(__name__)



def main(argv):
  pass

if __name__ == "__main__":
  try:
    sys.exit(main(sys.argv[1:]))
  except Exception as e:
    halraiser(e)