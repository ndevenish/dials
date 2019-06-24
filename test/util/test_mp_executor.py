from __future__ import absolute_import, division, print_function

import time
from concurrent.futures import wait

from dials.util.mp import BatchExecutor


def _do_long_function(i):
    """Do a long function. Has to be pickleable so in root"""
    print("Starting", i)
    time.sleep(i)
    print(i)
    return i * 2


def test_batch_processor_easy_mp_passthrough():
    with BatchExecutor("multiprocessing", max_workers=5, njobs=1) as e:
        futs = []
        start = time.time()
        for i in range(5):
            futs.append(e.submit(_do_long_function, i))
        wait(futs)
        result = [f.result() for f in futs]
        assert result == [0, 2, 4, 6, 8]
        duration = time.time() - start
        assert duration < 7

        start = time.time()
        result = e.map(_do_long_function, range(5))
        assert list(result) == [0, 2, 4, 6, 8]
        duration = time.time() - start
        assert duration < 7


# try:

# except KeyboardInterrupt:
#     pass

# breakpoint()
