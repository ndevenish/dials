#!/usr/bin/env python
# coding: utf-8

"""
Find the most appropriate laue group from symmetry-related integrated data.

Usage:
  dials.find_laue_group [options] <data_file>...
"""

from __future__ import absolute_import, print_function, division

import re
import sys
import logging
import itertools
from StringIO import StringIO
from math import log10, floor
from functools import reduce
import time

import enum

import dials.util.log
import dials.array_family.flex
from dials.util import halraiser
from dials.util.options import OptionParser, flatten_reflections, flatten_experiments

from dxtbx.model.experiment_list import ExperimentListFactory

from libtbx import phil

from cctbx import miller, sgtbx
import dials.algorithms.symmetry.origin as dials_origin


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

def all_space_groups():
  for x in range(1,230):
    sgbase = sgtbx.space_group_info(number=x)
    if str(sgbase.reference_setting()) == str(sgbase.primitive_setting()):
      yield sgbase.group()
    else:
      yield sgbase.reference_setting().group()
      yield sgbase.primitive_setting().group()

def all_compatible_space_groups(unit_cell,**kwargs):
  for group in all_space_groups():
    if group.is_compatible_unit_cell(unit_cell, **kwargs):
      yield group

def _get_reflections_filter(reflections):
  """Calculates and returns a reflections selection filter.

  This filters for spots that have been integrated, and validates that the 
  assumptions about the data make sense.
  """
  logger.debug("Filtering {} reflections:".format(len(reflections)))
  # Only use reflections that were actually integrated
  filter_integrated = reflections.get_flags(reflections.flags.integrated)
  logger.debug("  {} reflections were integrated ({} ignored)".format(sum(filter_integrated), len(reflections)-sum(filter_integrated)))
  # Validate that this filtered all the entries without an experiment
  assert not any((reflections["id"] < 0) & filter_integrated), "Experiment-less reflections have been integrated"
  # Check that the variances are sensible
  assert not any((reflections['intensity.sum.variance'] <= 0) & filter_integrated), "Integrated reflections should always have variance"

  return filter_integrated

def to_pandas(table):
  """Convert a reflection table to a pandas dataframe"""
  # Convert to a pandas dataframe for some probing
  import pandas as pd
  df = pd.DataFrame({y[0]: [x for x in table[y[0]]] for y in table.cols()})
  # Make flags easy-to-interpret
  df["flags"] = [Flags.resolve(x) for x in df["flags"]]
  return df

  # # Remove data with no miller index
  # accept_entries = df["miller_index"] != (0,0,0)
  # # Must have a non-zero variance
  # accept_entries &= df["intensity.sum.variance"] > 0

  # data = df[accept_entries]

  # h, k, l = [df["miller_index"].apply(lambda x: x[n]) for n in [0,1,2]]


  # logger.debug("Range on h: {:-2d}, {:-2d}".format(h.min(), h.max()))
  # logger.debug("Range on k: {:-2d}, {:-2d}".format(k.min(), k.max()))
  # logger.debug("Range on l: {:-2d}, {:-2d}".format(l.min(), l.max()))

def group_reflections(table, space_group=None):
  """Group equivalent reflections in a reflection table"""
  assert space_group is None
  start = time.time()

  avail_reflections = set(table["miller_index"])
  unique_reflections = set()
  while avail_reflections:
    reflection = avail_reflections.pop()
    unique_reflections.add(reflection)
    mate = tuple(-x for x in reflection)
    if mate in avail_reflections:
      avail_reflections.remove(mate)
  print("Down to {} unique reflections".format(len(unique_reflections)))

  results = {x: dials.array_family.flex.reflection_table() for x in unique_reflections}
  
  # for reflection in unique_reflections:
  for i, row in enumerate(table):
    if time.time() - start > 10:
      start = time.time()
      print("{} remaining".format(len(table)-i))

    reflection = row["miller_index"]
    if not row["miller_index"] in unique_reflections:
      reflection = tuple(-x for x in reflection)
    results[reflection].append(row)


  # # For now, ignore space group and merge +, -
  # # available = ~dials.array_family.flex.bool(len(reflections))
  # while avail_reflections:

  #   reflection = avail_reflections.pop()
  #   mate = tuple(-x for x in reflection)
  #   avail_reflections.remove(mate)
  #   matches = table["miller_index"] == reflection
  #   # Match friedel pairs
  #   matches |= table["miller_index"] == mate
  #   # Pull out the results, then reduce the table
  #   results[reflection] = table.select(matches)
  #   table = table.select(~matches)

  return results


def find_laue_group(experiments, reflections):
  """Do the main routine. Can be called separately from instantiation"""

  # For now, easier to verify that we don't have multiple crystals
  assert all(experiments[0].crystal == x.crystal for x in experiments[1:])

  # Filter each table down to the valid data
  reflections = [ref.select(_get_reflections_filter(ref)) for ref in reflections]
  # Flatten all tables into one megatable
  reflections = reduce(lambda x, y: x.append(y), reflections)

  # To copy check_indexing_symmetry's progress, print some hkl stats
  h, k, l = [[x[n] for x in reflections["miller_index"]] for n in range(3)]
  logger.debug("Range on h: {:-2d}, {:-2d}".format(min(h), max(h)))
  logger.debug("Range on k: {:-2d}, {:-2d}".format(min(k), max(k)))
  logger.debug("Range on l: {:-2d}, {:-2d}".format(min(l), max(l)))

  symmetry = dials_origin.cctbx_crystal_from_dials(experiments[0].crystal)
  oind = reflections["miller_index"].deep_copy()
  mset = miller.set(crystal_symmetry=symmetry, indices=reflections["miller_index"])

  # import pdb
  # pdb.set_trace()
  # list(all_compatible_space_groups(experiments[0].crystal.get_unit_cell(), relative_length_tolerance=0.3))

  import code
  code.interact(local=dict(globals(), **locals()))

  
def _load_experiment(filename, check_format=False):
  """Convenience method to load an experiments file from a filename"""
  return ExperimentListFactory.from_json_file(filename, check_format=check_format)

def _load_reflections(filename):
  return dials.array_family.flex.reflection_table.from_pickle(filename)

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

  # UNWRAP all of the data objects from the PHIL parser
  all_reflections = flatten_reflections(params.input.reflections)
  all_experiments = flatten_experiments(params.input.experiments)
  
  find_laue_group(experiments=all_experiments, reflections=all_reflections)

##############################################################################
# Monkeypatching scitbx/cctbx for diagnostics output
#
# For safety and sanity, nothing is applied unless we run this module as __main__

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
  def __repr__(self):
    flag_names = sorted([n.name.upper() for n in self]) # Flags.resolve(self.value)
    if flag_names:
      return " | ".join(flag_names)
    else:
      return "{None}"

# Remove dtype= section from numpy output
re_remove_dtype = re.compile(r"(?:,|\()\s*dtype=\w+(?=,|\))")
# Regular expression to help make repr's oneline
reReprToOneLine = re.compile("\n\\s+")

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
  s = type(self).__name__ + "("
  if len(self):
    s += "["

    indent = ",\n"+ " "*len(s)
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

    s += "]"
  s += ")"
  return s

def _double_vec_repr(self):
  """Special-case repr for miller-index objects"""
  s = type(self).__name__ + "("
  if self:
    s += "["
    indent = "\n"+ " "*len(s)

    if len(self) > _summaryThreshold:
      "({: 3d}, {: 3d}, {: 3d})"
      s += indent.join(repr(x) for x in self[:_summaryEdgeItems])
      s += indent + "..." + indent
      s += indent.join(repr(x) for x in self[-_summaryEdgeItems:])
    else:
      s += indent.join(repr(x) for x in self)
    s += "]"
  s += ")"
  return s

_max_column_width = 50
_max_column_height = 60

def _reftable_repr(self):
  _max_display_width = 100
  s = "<{}".format(type(self).__name__)
  if self:
    s += "\n"
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
  s += ">"
  if self:
    s += "\n[{} rows x {} columns]".format(len(self), len(list(self.keys())))
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

def _cctbx_crystal_symmetry_repr(self):
  parts = []
  if self.space_group_info() is not None:
    parts.append("space_group_symbol='{}'".format(str(self.space_group_info())))
  if self.unit_cell() is not None:
    parts.append("unit_cell={}".format(self.unit_cell().parameters()))
  return "symmetry({})".format(", ".join(parts))

def _cctbx_miller_set_repr(self):
  """Repr function for miller set and array objects"""
  parts = ["crystal_symmetry=" + _cctbx_crystal_symmetry_repr(self)]
  if self.indices():
    parts.append("indices="+ reReprToOneLine.sub(" ", repr(self.indices())))
  if self.anomalous_flag() is not None:
    parts.append("anomalous_flag=" + str(self.anomalous_flag()))
  if hasattr(self, "data") and self.data() is not None:
    parts.append("data="+ reReprToOneLine.sub(" ", repr(self.data())))
  if hasattr(self, "sigmas") and self.sigmas() is not None:
    parts.append("sigmas="+ reReprToOneLine.sub(" ", repr(self.sigmas())))
  if hasattr(self, "info") and self.info() is not None:
    parts.append("sigmas="+ reReprToOneLine.sub(" ", repr(self.sigmas())))

  return type(self).__name__ + "(" + ", ".join(parts) + ")"

def do_monkeypatching():
  import scitbx.array_family.flex
  import cctbx.array_family.flex
  import cctbx.uctbx
  import dxtbx.model
  import cctbx.crystal
  import cctbx.sgtbx
  import cctbx.miller

  _patch_flex(scitbx.array_family.flex.size_t, int)
  _patch_flex(scitbx.array_family.flex.double, float)
  _patch_flex(scitbx.array_family.flex.int, int)
  _patch_flex(scitbx.array_family.flex.bool, bool)
  
  cctbx.array_family.flex.miller_index.__repr__ = _miller_repr
  scitbx.array_family.flex.vec3_double.__repr__ = _double_vec_repr
  dials.array_family.flex.reflection_table.__repr__ = _reftable_repr

  phil.scope_extract.__repr__ = _phil_repr
  phil.scope_extract.__str__ = lambda x: x.__repr__(in_scope=True)

  cctbx.crystal.symmetry.__repr__ = _cctbx_crystal_symmetry_repr
  cctbx.miller.set.__repr__ = _cctbx_miller_set_repr
  cctbx.sgtbx.space_group_info.__repr__ = lambda x: type(x).__name__ + "('{}')".format(str(x))
  cctbx.sgtbx.space_group.__repr__ = lambda x: "<" + type(x).__name__ + " type=" + repr(x.type().lookup_symbol()) + ">"

  cctbx.uctbx.unit_cell.__repr__ = lambda x: "{}({})".format(type(x).__name__, x.parameters())
  cctbx.uctbx.ext.unit_cell.__repr__ = lambda x: "{}({})".format(type(x).__name__, x.parameters())

  dxtbx.model.ExperimentList.__repr__ = lambda self: "[" + ", ".join(repr(x) for x in self) + "]"

  # <cctbx_uctbx_ext.unit_cell object at 0x101077c20>

if __name__ == "__main__":
  try:
    # Monkeypatching - only if run as a __main__
    do_monkeypatching()

    sys.exit(main(sys.argv[1:]))
  except Exception as e:
    halraiser(e)

