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
from StringIO import StringIO

import dials.util.log
from dials.util import halraiser
from dials.util.options import OptionParser, flatten_reflections, flatten_experiments

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

def _phil_repr(self, in_scope=False):
  """Hack in a phil.scope_extract repr function"""
  s = StringIO()
  if not in_scope:
    s.write('<phil.scope_extract """')
  s.write('{\n')
  # Step over every named element in this
  phil_names = sorted([x for x in self.__dict__ if not x.startswith("_")])
  # Get the maximum length of an attribute (non sub-scope) value
  max_len = max(len(x) for x in phil_names if not isinstance(getattr(self, x), phil.scope_extract))
  for name in phil_names:
    s.write("  " + name.ljust(max_len) + " ")
    value = getattr(self, name)

    if isinstance(value, phil.scope_extract):
      # Get the representation, then add an indentation to every line
      subscope = value.__repr__(in_scope=True)
      subscope = "\n".join("  " + x for x in subscope.splitlines()).strip()
      s.write(subscope)
    else:
      # Just output the value
      s.write("= " + repr(value))
    s.write("\n")
  s.write('}')
  if not in_scope:
    s.write('""">')
  return s.getvalue()

def find_laue_group(experiments, reflections):
  """Do the main routine. Can be called separately from instantiation"""


def main(argv):
  optionparser = OptionParser(
    usage=__doc__.strip(),
    read_experiments=True,
    read_reflections=True,
    read_datablocks=False,
    phil=phil_scope,
    check_format=False)
  params, options = optionparser.parse_args(argv)

  # For now, disable the file logging
  # dials.util.log.config(
  #   verbosity=options.verbose,
  #   info=params.output.log,
  #   debug=params.output.debug_log)
  dials.util.log.config(verbosity=5)

  if not params.input.experiments or not params.input.reflections:
    optionparser.print_help()
    return

  logger.debug("Parameters = {}".format(params))

  all_reflections = flatten_reflections(params.input.reflections)
  #[x.data for x in params.input.reflections]
  all_experiments = flatten_experiments(params.input.experiments)
  #list(itertools.chain(*[x.data for x in params.input.experiments]))

  find_laue_group(experiments=all_experiments, reflections=all_reflections)

  # # Convert to a pandas dataframe for some probing
  # print("Converting")
  # import pandas as pd
  # ref = all_reflections[0]
  # df = pd.DataFrame({y[0]: [x for x in ref[y[0]]] for y in ref.cols()})
  # store = pd.HDFStore('reflections.h5')
  # store["reflections"] = df
  # store.close()

  import pdb
  pdb.set_trace()
  

if __name__ == "__main__":
  try:
    # Monkeypatching - only if run as a __main__
    phil.scope_extract.__repr__ = _phil_repr
    phil.scope_extract.__str__ = lambda x: x.__repr__(in_scope=True)
  
    sys.exit(main(sys.argv[1:]))
  except Exception as e:
    halraiser(e)
