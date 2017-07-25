#!/usr/bin/env python
# coding: utf-8

"""
Find the most appropriate laue group from symmetry-related integrated data.

Usage:
  dials.find_laue_group [options] <data_file>...
"""

from __future__ import absolute_import, print_function, division

import sys
import logging
import itertools

import dials.util.log
from dials.util import halraiser
from dials.util.options import OptionParser

from libtbx import phil

logger = logging.getLogger("dials.find_laue_group")

phil_scope = phil.parse('''
  debug = False
    .type = bool
    .help = "Output additional debugging information"

  output {
    log = dials.find_laue_group.log
      .type = path
      .help = "The log filename"

    debug_log = dials.find_laue_group.debug.log
      .type = path
      .help = "The debug log filename"
  }
''')


def main(argv):
  optionparser = OptionParser(
    usage=__doc__.strip(),
    read_experiments=True,
    read_reflections=True,
    read_datablocks=False,
    phil=phil_scope)
  params, options = optionparser.parse_args(argv)

  dials.util.log.config(
    verbosity=options.verbose,
    info=params.output.log,
    debug=params.output.debug_log)


  all_reflections = [x.data for x in params.input.reflections]
  all_experiments = list(itertools.chain(*[x.data for x in params.input.experiments]))

  # Convert to a pandas dataframe
  # pd.DataFrame({y[0]: [x for x in ref[y[0]]] for y in ref.cols()})

  import pdb
  pdb.set_trace()
  

if __name__ == "__main__":
  try:
    sys.exit(main(sys.argv[1:]))
  except Exception as e:
    halraiser(e)
