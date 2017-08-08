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
from math import log10, floor

import dials.util.log
import dials.array_family.flex
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

import enum
class Flags(enum.IntEnum):
  BACKGROUND_INCLUDES_BAD_PIXELS = 32768
  BAD_REFERENCE = 2097152
  BAD_SPOT = 64512
  CENTROID_OUTLIER = 131072
  DONT_INTEGRATE = 128
  FAILED_DURING_BACKGROUND_MODELLING = 262144
  FAILED_DURING_PROFILE_FITTING = 1048576
  FAILED_DURING_SUMMATION = 524288
  FOREGROUND_INCLUDES_BAD_PIXELS = 16384
  IN_POWDER_RING = 8192
  INCLUDES_BAD_PIXELS = 49152
  INDEXED = 4
  INTEGRATED = 768
  INTEGRATED_PRF = 512
  INTEGRATED_SUM = 256
  OBSERVED = 2
  OVERLAPPED_BG = 2048
  OVERLAPPED_FG = 4096
  OVERLOADED = 1024
  PREDICTED = 1
  REFERENCE_SPOT = 64
  STRONG = 32
  USED_IN_MODELLING = 65536
  USED_IN_REFINEMENT = 8

  @classmethod
  def resolve(cls, value):
    return FlagViewer({ev for ev in cls if value & ev.value})

class FlagViewer(set):
  # def __init__(self, value):
  #   self.value = value
  def __repr__(self):
    flag_names = sorted([n.name.upper() for n in self]) # Flags.resolve(self.value)
    if flag_names:
      return " | ".join(flag_names)
    else:
      return "{None}"
  # def __hash__(self):
  #   return self.value

def find_laue_group(experiments, reflections):
  """Do the main routine. Can be called separately from instantiation"""

  # Convert to a pandas dataframe for some probing
  print("Converting")
  import pandas as pd
  ref = reflections[0]


  import code
  # code.interact(local=locals())
  code.interact(local=dict(globals(), **locals()))



  df = pd.DataFrame({y[0]: [x for x in ref[y[0]]] for y in ref.cols()})
  # Make flags easy-to-interpret
  df["flags"] = [Flags.resolve(x) for x in df["flags"]]

  # Remove data with no miller index
  accept_entries = df["miller_index"] != (0,0,0)
  # Must have a non-zero variance
  accept_entries &= df["intensity.sum.variance"] > 0

  data = df[accept_entries]

  h, k, l = [df["miller_index"].apply(lambda x: x[n]) for n in [0,1,2]]


  logger.debug("Range on h: {:-2d}, {:-2d}".format(h.min(), h.max()))
  logger.debug("Range on k: {:-2d}, {:-2d}".format(k.min(), k.max()))
  logger.debug("Range on l: {:-2d}, {:-2d}".format(l.min(), l.max()))

  import code
  # code.interact(local=locals())
  code.interact(local=dict(globals(), **locals()))


  


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

  # Unwrap all of the data objects from the PHIL parser
  all_reflections = flatten_reflections(params.input.reflections)
  all_experiments = flatten_experiments(params.input.experiments)
  
  find_laue_group(experiments=all_experiments, reflections=all_reflections)

##############################################################################
# Monkeypatching scitbx/cctbx for diagnostics output
#
# For safety and sanity, nothing is applied unless we run this module as __main__

import re

re_remove_dtype = re.compile(r"(?:,|\()\s*dtype=\w+(?=,|\))")

_summaryEdgeItems = 3     # repr N leading and trailing items of each dimension
_summaryThreshold = 1000  # total items > triggers array summarization

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

def _miller_repr(self):
  """Special-case repr for miller-index objects"""
  s = type(self).__name__ + "(["
  indent = "\n"+ " "*len(s)
  # Work out how to align the data
  format_sample = self
  if len(self) > _summaryThreshold:
    format_sample = list(self[:_summaryEdgeItems]) + list(self[-_summaryEdgeItems:])
  # Do we have negative symbols
  negs = [any(x[i] < 0 for x in format_sample) for i in range(3)]
  # Maximum width
  maxw = [max(int(1+floor(log10(abs(x[i])) if x[i] != 0 else 0)) for x in format_sample) for i in range(3)]
  fmts = "(" + ", ".join(["{{:{}{}d}}".format(" " if neg else "", w+(1 if neg else 0)) for neg, w in zip(negs, maxw)]) + ")"

  # tup_fmt = ()

  if len(self) > _summaryThreshold:
    "({: 3d}, {: 3d}, {: 3d})"
    s += indent.join(fmts.format(*x) for x in self[:_summaryEdgeItems])
    s += indent + "..." + indent
    s += indent.join(fmts.format(*x) for x in self[-_summaryEdgeItems:])
  else:
    s += indent.join(fmts.format(*x) for x in self)
  s += "])"
  return s

def _double_vec_repr(self):
  """Special-case repr for miller-index objects"""
  s = type(self).__name__ + "(["
  indent = "\n"+ " "*len(s)

  if len(self) > _summaryThreshold:
    "({: 3d}, {: 3d}, {: 3d})"
    s += indent.join(repr(x) for x in self[:_summaryEdgeItems])
    s += indent + "..." + indent
    s += indent.join(repr(x) for x in self[-_summaryEdgeItems:])
  else:
    s += indent.join(repr(x) for x in self)
  s += "])"
  return s

_max_column_width = 50
_max_column_height = 60

def _reftable_repr(self):
  _max_display_width = 100
  s = "{}(\n".format(type(self).__name__)
  indent = "    "
  maxcol = max(len(x) for x in self.keys())

  rows = []
  for column in sorted(self.keys()):
    row = indent + column.ljust(maxcol) + " = "
    # Now do a single-line representation of the column....
    data = self[column]
    remaining_space = _max_display_width - len(row)
    data_repr = " ".join(x.strip() for x in repr(data).splitlines())
    if len(data_repr) > remaining_space:
      data_repr = data_repr[:remaining_space-3] + "..."
    row += data_repr
    rows.append(row)
  s += "\n".join(rows)
  s += ")\n"
  s += "[{} rows x {} columns]".format(len(self), len(list(self.keys())))
  return s

# re_remove_dtype
def _patch_flex(flex, dtype, shape=None, ndim=1):
  import numpy
  flex.__repr__ = lambda x: re_remove_dtype.sub("", numpy.array_repr(x))
  if shape is None:
    flex.shape = property(lambda x: (len(x),))
  else:
    flex.shape = property(shape)
  flex.ndim = ndim
  flex.dtype = numpy.dtype(dtype)
  # dtype('int64')

def do_monkeypatching():
  import scitbx.array_family.flex
  import cctbx.array_family.flex
  # import dials.array_family.flex

  _patch_flex(scitbx.array_family.flex.size_t, int)
  _patch_flex(scitbx.array_family.flex.double, float)
  _patch_flex(scitbx.array_family.flex.int, int)
  _patch_flex(scitbx.array_family.flex.bool, bool)
  
  cctbx.array_family.flex.miller_index.__repr__ = _miller_repr
  scitbx.array_family.flex.vec3_double.__repr__ = _double_vec_repr
  dials.array_family.flex.reflection_table.__repr__ = _reftable_repr

  phil.scope_extract.__repr__ = _phil_repr
  phil.scope_extract.__str__ = lambda x: x.__repr__(in_scope=True)

if __name__ == "__main__":
  try:
    # Monkeypatching - only if run as a __main__
    do_monkeypatching()

    sys.exit(main(sys.argv[1:]))
  except Exception as e:
    halraiser(e)

