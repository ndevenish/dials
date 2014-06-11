#
# masker.py
#
#  Copyright (C) 2013 Diamond Light Source
#
#  Author: James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

from __future__ import division

class Masker(object):
  '''A class to perform all the shoebox masking.'''

  def __init__(self, experiment, delta_d=None, delta_m=None):
    ''' Initialise the masking algorithms

    Params:
        experiment The experiment data
        delta_d The extent of the reflection in reciprocal space
        delta_m The extent of the reflection in reciprocal space

    '''
    from dials.algorithms.shoebox import MaskOverlapping
    from dials.algorithms.shoebox import MaskForeground

    # Construct the overlapping reflection mask
    self.mask_overlapping = MaskOverlapping()

    if delta_d is None or delta_m is None:
      self.mask_foreground = None
    else:
      # Construct the foreground pixel mask
      self.mask_foreground = MaskForeground(
          experiment.beam, experiment.detector,
          experiment.goniometer, experiment.scan,
          delta_d, delta_m)

  def __call__(self, reflections, adjacency_list=None):
    ''' Mask the given reflections.

    Params:
        reflections The reflection list
        adjacency_list The adjacency_list (optional)

    Returns:
        The masked reflection list

    '''
    from dials.util.command_line import Command

    # Mask the overlaps if an adjacency list is given
    if adjacency_list:
      Command.start('Masking overlapping reflections')
      self.mask_overlapping(
        reflections['shoebox'],
        reflections['xyzcal.px'],
        adjacency_list)
      Command.end('Masked {0} overlapping reflections'.format(
          len(adjacency_list)))

    if self.mask_foreground:
      # Mask the foreground region
      Command.start('Masking foreground pixels')
      self.mask_foreground(
        reflections['shoebox'],
        reflections['s1'],
        reflections['xyzcal.px'].parts()[2])
      Command.end('Masked foreground pixels for {0} reflections'.format(
        len(reflections)))

    # Return the reflections
    return reflections
