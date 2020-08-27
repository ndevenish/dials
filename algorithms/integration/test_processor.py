from __future__ import absolute_import, division, print_function

import math

from dials.array_family import flex
from dials_algorithms_integration_integrator_ext import JobList
from dials.algorithms.profile_model.gaussian_rs import Model
from dxtbx.model.experiment_list import ExperimentListFactory
import dials.algorithms.integration.processor
import mock
import pytest


def test_shoebox_memory_is_a_reasonable_guesstimate(dials_data):
    path = dials_data("centroid_test_data").join("experiments.json").strpath

    exlist = ExperimentListFactory.from_json_file(path)[0]
    exlist.profile = Model(
        None, n_sigma=3, sigma_b=math.radians(0.024), sigma_m=math.radians(0.044),
    )

    rlist = flex.reflection_table.from_predictions(exlist)
    rlist["id"] = flex.int(len(rlist), 0)
    rlist["bbox"] = flex.int6(rlist.size(), (0, 1, 0, 1, 0, 1))

    jobs = JobList()
    jobs.add((0, 1), (0, 9), 9)
    for flatten in (True, False):
        assumed_memory_usage = list(jobs.shoebox_memory(rlist, flatten))
        assert len(assumed_memory_usage) == 1
        assert assumed_memory_usage[0] == pytest.approx(23952, abs=3000)


@mock.patch("dials.algorithms.integration.processor.flex.max")
@mock.patch("dials.algorithms.integration.processor.psutil.virtual_memory")
@mock.patch("dials.algorithms.integration.processor.psutil.swap_memory")
def test_runtime_error_raised_when_not_enough_memory(
    mock_psutil_swap, mock_psutil_vm, mock_flex_max
):
    mock_flex_max.return_value = 750001
    mock_psutil_vm.return_value.available = 1000000
    mock_psutil_swap.return_value.free = 0

    phil_mock = mock.Mock()
    phil_mock.mp.method = "multiprocessing"
    phil_mock.mp.nproc = 4
    phil_mock.block.max_memory_usage = 0.75

    reflections = {"bbox": flex.int6(1000, (0, 1, 0, 1, 0, 1))}
    manager = dials.algorithms.integration.processor._Manager(
        None, reflections, phil_mock
    )
    manager.jobs = mock.Mock(autospec=JobList)

    with pytest.raises(MemoryError) as exc_info:
        manager.compute_processors()
    assert "Not enough memory to run integration jobs." in exc_info.value.args[0]
    mock_flex_max.assert_called_once_with(manager.jobs.shoebox_memory.return_value)

    # Reduce memory usage by 1 byte, should then pass
    mock_flex_max.return_value = 750000
    manager.compute_processors()
    mock_flex_max.assert_called_with(manager.jobs.shoebox_memory.return_value)
