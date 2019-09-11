from __future__ import absolute_import, division, print_function

from cctbx import sgtbx, uctbx
from scitbx import matrix


def get_inverse_ub_matrix_from_xparm(handle):
    """Get the inverse_ub_matrix from an xparm file handle

    Params:
        handle The file handle

    Returns:
        The inverse_ub_matrix
    """
    return matrix.sqr(
        handle.unit_cell_a_axis + handle.unit_cell_b_axis + handle.unit_cell_c_axis
    )


def get_ub_matrix_from_xparm(handle):
    """Get the ub_matrix from an xparm file handle

    Params:
        handle The file handle

    Returns:
        The ub_matrix
    """
    return get_inverse_ub_matrix_from_xparm(handle).inverse()


def get_unit_cell_from_xparm(handle):
    """Get the unit_cell object from an xparm file handle

    Params:
        handle The file handle

    Returns:
        The unit cell object
    """
    return uctbx.unit_cell(parameters=handle.unit_cell)


def get_space_group_type_from_xparm(handle):
    """Get the space group tyoe object from an xparm file handle

    Params:
        handle The file handle

    Returns:
        The space group type object
    """
    return sgtbx.space_group_type(
        sgtbx.space_group(sgtbx.space_group_symbols(handle.space_group).hall())
    )
