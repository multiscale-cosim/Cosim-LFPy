#!/usr/bin/env python
# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multiscale Simulation and Design
#
# ------------------------------------------------------------------------------
import sys

from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import enums
from ms_manager import MSManager


def main():
    """
    :desc: Entry point for Co-Simulation CoSimulator tool
    :param args: user command line arguments
    :return: CoSimulator's return code to be used as exit code by the bash environment
    """
    ms_manager = MSManager()
    ms_manager_rc = ms_manager.run()

    if ms_manager_rc == enums.CoSimulatorReturnCodes.OK:
        # finished properly!
        return enums.BashReturnCodes.SUCCESSFUL  # 0
    # something went wrong
    elif ms_manager_rc == enums.CoSimulatorReturnCodes.PARAMETER_ERROR:
        print('ERROR: commandline argument parsing!')
        return enums.BashReturnCodes.CO_SIMULATOR_PARAMETER_ERROR
    elif ms_manager_rc == enums.CoSimulatorReturnCodes.VARIABLE_ERROR:
        return enums.BashReturnCodes.CO_SIMULATOR_VARIABLE_ERROR
    elif ms_manager_rc == enums.CoSimulatorReturnCodes.XML_ERROR:
        return enums.BashReturnCodes.CO_SIMULATOR_XML_ERROR
    elif ms_manager_rc == enums.CoSimulatorReturnCodes.LAUNCHER_ERROR:
        return enums.BashReturnCodes.LAUNCHER_ERROR
    else:
        return enums.BashReturnCodes.CO_SIMULATOR_ERROR


if __name__ == '__main__':
    sys.exit(main())
