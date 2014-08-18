#!/usr/bin/env python
# -*- mode: python; coding: utf-8; indent-tabs-mode: nil; python-indent: 2 -*-
#
# dials.algorithms.indexing.known_orientation.py
#
#  Copyright (C) 2014 Diamond Light Source
#
#  Author: Richard Gildea
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

from __future__ import division

from dials.algorithms.indexing.indexer import indexer_base
from dxtbx.model.experiment.experiment_list import Experiment, ExperimentList

class indexer_known_orientation(indexer_base):

  def __init__(self, reflections, sweep, params, known_orientations):
    self.known_orientations = known_orientations
    super(indexer_known_orientation, self).__init__(reflections, sweep, params)

  def find_lattices(self):
    experiments = ExperimentList()
    for cm in self.known_orientations:
      # indexer expects crystals to be in primitive setting
      space_group = cm.get_space_group()
      cb_op_to_primitive \
        = space_group.info().change_of_basis_op_to_primitive_setting()
      cm = cm.change_basis(cb_op_to_primitive)
      experiments.append(Experiment(beam=self.beam,
                                    detector=self.detector,
                                    goniometer=self.goniometer,
                                    scan=self.imagesets[0].get_scan(),
                                    crystal=cm))
    return experiments
