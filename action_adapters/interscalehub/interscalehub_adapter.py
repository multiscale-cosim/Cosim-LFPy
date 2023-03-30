# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------

import sys
import time
import pickle
import base64

from common.utils.security_utils import check_integrity
from action_adapters_alphabrunel.setup_result_directories import SetupResultDirectories
from EBRAINS_InterscaleHUB.Interscale_hub.manager_nest_to_tvb import NestToTvbManager
from EBRAINS_InterscaleHUB.Interscale_hub.manager_tvb_to_nest import TvbToNestManager
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_EXCHANGE_DIRECTION 
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.configurations_manager import ConfigurationsManager

'''
TODO Refactor to fit NEST LFPy usecase
TODO Refactor to read the parameters from science/parameters
TODO add new enums for directions e.g. nest to lfpy
TODO change the directions enums as e.g. receive from NEST, Send to NEST, etc.
'''


def run_wrapper(direction, configurations_manager, log_settings,
                sci_params_xml_path_filename=''):
    """
    starts the inter-scale hub to exchange data in direction provided as the
    parameter.
    """
    # direction
    # 1 --> nest to Tvb
    # 2 --> tvb to nest
    # NOTE hardcoded path
    # TODO get the path as an argument from launcher
    path = configurations_manager.get_directory(
                                        directory=DefaultDirectories.SIMULATION_RESULTS)
    # parameters = Parameter(path)
    # NOTE will be changed after refactoring
    parameters = {
                "path": path,
                # "simulation_time": 1000.0,
                # "level_log": 1,
                "id_nest_region": [0],  # TO BE DONE: supporting Python's list on XML sci-params
                'id_first_neurons': [1],
                # "save_spikes": True,
                # "save_rate": True,
                "width": 20.0,
                "id_first_spike_detector": 229
        }

    direction = int(direction)  # NOTE: will be changed

    # Case a: Nest to TVB inter-scale hub
    import socket
    if direction == DATA_EXCHANGE_DIRECTION.NEST_TO_LFPy:
        # create directories to store parameter.json file, 
        # port information, and logs
        # print(f"__DEBUG__ NEST_TO_TVB *** host_name: {socket.gethostname()}")
        SetupResultDirectories(path)  # NOTE: will be changed
        hub = NestToTvbManager(parameters,
                               configurations_manager,
                               log_settings,
                               sci_params_xml_path_filename=sci_params_xml_path_filename)

    # 1) init steering command
    # includes param setup, buffer creation
    # NOTE init is system action and so is done implicitly with the hub
    # initialization

    # 2) Start steering command
    # receive, pivot, transform, send
    hub.start()

    # 3) Stop steering command
    # disconnect and close ports
    hub.stop()

if __name__ == '__main__':
    # RunSetup()
    direction = sys.argv[1]
    configurations_manager = pickle.loads(base64.b64decode(sys.argv[2]))
    log_settings = pickle.loads(base64.b64decode(sys.argv[3]))
    
    # security check of pickled objects
    # it raises an exception, if the integrity is compromised
    check_integrity(configurations_manager, ConfigurationsManager)
    check_integrity(log_settings, dict)
    # everything is fine, run InterscaleHub
    sys.exit(run_wrapper(direction,
                         configurations_manager,
                         log_settings,
                         sci_params_xml_path_filename=sys.argv[4]))