# -----------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
# -----------------------------------------------------------------------------
import subprocess
import fcntl
import os
import pickle
import base64
import time

from EBRAINS_RichEndpoint.application_companion.common_enums import Response


def b64encode_and_pickle(logger, obj):
        """
        helper function to encode base64 and pickle a given (picklable)
        object
        """
        try:
            encoded_pickled_obj = base64.b64encode(pickle.dumps(obj))
            logger.debug(f"{obj} is encoded and pickled: {encoded_pickled_obj}")
            return encoded_pickled_obj
        except pickle.PicklingError:
            logger.exception(f"could not pickled {obj}!")
            return Response.ERROR
        except ValueError:
            logger.exception(f"could not encode {obj}!")
            return Response.ERROR
        except TypeError:
            logger.exception(f"could not encode {obj}!")
            return Response.ERROR


def non_block_read(logger, std_stream):
    """
    helper function for reading from output/error stream of the process
    launched.
    """
    fd = std_stream.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        return std_stream.read()
    except Exception:
        logger.exception(
            f'exception while reading from {std_stream}')
        return ''


def terminate_with_error_loudly(logger, error_summary):
    """
    Logs the exception with traceback and returns with ERROR as response to
    terminate with error"""
    try:
        # raise RuntimeError exception
        raise RuntimeError
    except RuntimeError:
        # log the exception with traceback
        logger.exception(error_summary)
    # respond with Error to terminate
    return Response.ERROR


def stop_preemptory(logger, process):
    """helper function to terminate the application forcefully."""
    logger.critical("terminating preemptory")
    logger.info(f"going to signal PID={process.pid} to terminate.")
    process.terminate()
    time.sleep(0.001)  # let the Popen process be finished
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        # Could not terminate the process,
        # send signal to forcefully kill it
        logger.info(f"going to signal "
                            f"PID={process.pid} "
                            f"to forcefully quit.")
        # quit the process forcefully
        process.kill()

    # check whether the process finishes
    exit_status = process.poll()
    # Worst Case, process could not be terminated/killed
    if exit_status is None:
        return terminate_with_error_loudly(logger,
            "could not terminate the process "
            f"PID={process.pid}")

    # Case, process is terminated/killed
    logger.info(f"terminated PID={process.pid}"
                f" exit_status={exit_status}")
    return Response.OK
