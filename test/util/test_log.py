from __future__ import absolute_import, division, print_function

import logging

import dials.util.log


def test_LoggingContext():

    # configure logging
    dials.util.log.config(verbosity=2)

    # get some loggers
    idx_logger = logging.getLogger("dials.algorithms.indexing")
    dials_logger = logging.getLogger("dials")

    # check the logging level is as expected
    assert idx_logger.getEffectiveLevel() == logging.DEBUG
    assert dials_logger.getEffectiveLevel() == logging.DEBUG

    # selectively change logging level and check this has worked
    with dials.util.log.LoggingContext(idx_logger, logging.ERROR):
        assert idx_logger.getEffectiveLevel() == logging.ERROR
        assert dials_logger.getEffectiveLevel() == logging.DEBUG

    # check logging levels are as they were before
    assert idx_logger.getEffectiveLevel() == logging.DEBUG
    assert dials_logger.getEffectiveLevel() == logging.DEBUG

    # now check we can pass logger as a string
    with dials.util.log.LoggingContext("dials.algorithms.indexing", logging.WARNING):
        assert idx_logger.getEffectiveLevel() == logging.WARNING
        assert dials_logger.getEffectiveLevel() == logging.DEBUG

    # check logging levels are as they were before
    assert idx_logger.getEffectiveLevel() == logging.DEBUG
    assert dials_logger.getEffectiveLevel() == logging.DEBUG


def test_lazyevaluate():
    calls = []
    values = [1]

    def _lazy_function():
        """Register the call and return the dynamic value"""
        calls.append(values[-1])
        return values[-1]

    l = dials.util.log.LazyEvaluate(_lazy_function)
    assert not calls
    assert str(l) == "1"
    assert calls == [1]
    # Check that the result is cached
    values.append(2)
    assert int(l) == 1
    assert str(l) == "1"
    assert float(l) == 1.0
    assert calls == [1]
