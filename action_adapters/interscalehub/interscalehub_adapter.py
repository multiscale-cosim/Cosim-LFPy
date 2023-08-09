# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------
import os
import sys
import time
import pickle
import base64
import ast

from mpi4py import MPI

from action_adapters.resource_usage_monitor_adapter import ResourceMonitorAdapter

from EBRAINS_Launcher.common.utils.security_utils import check_integrity
from EBRAINS_InterscaleHUB.Interscale_hub.manager_nest_to_lfpy import NestToLFPyManager
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_EXCHANGE_DIRECTION
from EBRAINS_RichEndpoint.application_companion.common_enums import SteeringCommands, COMMANDS, Response 
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.configurations_manager import ConfigurationsManager


class InterscaleHubAdapter:
    def __init__(self, direction,
                 configurations_manager,
                 log_settings,
                 is_monitoring_enabled,
                 sci_params_xml_path_filename=''):
        
        self.__log_settings = log_settings
        self.__configurations_manager = configurations_manager
        self.__logger = self.__configurations_manager.load_log_configurations(
            name="Application_Manager",
            log_configurations=self.__log_settings)
        
        # NOTE hardcoded path
        # TODO get the path as an argument from launcher
        self.__path = configurations_manager.get_directory(
            directory=DefaultDirectories.SIMULATION_RESULTS)
        
        self.__is_monitoring_enabled = is_monitoring_enabled
        self.__sci_params_xml_path = sci_params_xml_path_filename
        self.__parameters = self.__setup_parameters()
        self.__direction = int(direction)  # NOTE: will be changed
        self.__hub_name = None
        self.__hub = None
        self.__init_hub()
        self.__resource_usage_monitor = None
        # MPI rank
        self.__comm = MPI.COMM_WORLD
        self.__rank = self.__comm.Get_rank()
        self.__my_pid = os.getpid()
        self.__logger.debug(f"size: {self.__comm.Get_size()}, "
                            f"my rank: {self.__rank}, "
                            f"host_name:{os.uname()}")

        self.__logger.debug("initialized")

    @property
    def rank(self):
        return self.__rank

    @property
    def pid(self):
        return self.__my_pid

    @property
    def comm(self):
        return self.__comm

    def __setup_parameters(self):
        """returns parameters dictionary"""
        # NOTE hardcoded parameters
        # TODO get the parameters from parameters file
        return {
            "path": self.__path,
            "id_nest_region": [0],
            'id_first_neurons': [1],
            "width": 20.0,
            }

    def __init_hub(self):
        """Initializes InterscaleHub Manager object"""
        # Case a: Nest to LFPy inter-scale hub
        if self.__direction == DATA_EXCHANGE_DIRECTION.NEST_TO_LFPY:
            # create directories to store parameter.json file, 
            # port information, and logs
            # SetupResultDirectories(self.__path)  # NOTE: will be changed
            self.__hub_name = "NEST_TO_LFPy"
            self.__hub = NestToLFPyManager(
                self.__parameters,
                self.__configurations_manager,
                self.__log_settings,
                self.__direction,
                sci_params_xml_path_filename=self.__sci_params_xml_path)
        
        # Case b, direction is unknown for the current usecase
        # NOTE will be changed to usecase agnostic with InterscaleHub Refactoring
        else:
            # raise runtime exception
            try:
                raise RuntimeError
            except Exception:
                # log the exception with traceback
                self.__logger.exception('unknown InterscaleHub type: '
                                        f'{self.__direction}')
            # return with with error
            return Response.ERROR
            

        self.__logger.debug(f"initialized {self.__hub}")

    def execute_init_command(self):
        """executes INIT steering command"""
        self.__logger.debug("executing INIT command")
        # buffer setup and other initization is already done implicitly
        # start monitoring if enabled
        if self.__is_monitoring_enabled:
            self.__resource_usage_monitor = ResourceMonitorAdapter(
                configurations_manager,
                log_settings,
                os.getpid(),
                f"InterscaleHub_{self.__hub_name}")
            self.__resource_usage_monitor.start_monitoring()

        self.__logger.debug("INIT command is executed")

    def execute_start_command(self, spike_detectors_ids):
        """executes START steering command"""
        self.__logger.debug("executing START command")

        self.__hub.start(spike_detectors_ids)

        self.__logger.debug("START command is executed")

    def execute_end_command(self):
        """executes END steering command"""
        self.__logger.debug("executing END command")
        # stop hub
        self.__hub.stop()
        # stop monitoring if enabled
        if self.__is_monitoring_enabled:
            self.__resource_usage_monitor.stop_monitoring()

        self.__logger.debug("END command is executed")


if __name__ == '__main__':
    # TODO better handling of arguments parsing
    if len(sys.argv) == 6:
        direction = sys.argv[1]
        configurations_manager = pickle.loads(base64.b64decode(sys.argv[2]))
        log_settings = pickle.loads(base64.b64decode(sys.argv[3]))
        # get science parameters XML file path
        p_sci_params_xml_path_filename = sys.argv[4]
        # flag indicating whether resource usage monitoring is enabled
        is_monitoring_enabled = pickle.loads(base64.b64decode(sys.argv[5]))

        # security check of pickled objects
        # it raises an exception, if the integrity is compromised
        check_integrity(configurations_manager, ConfigurationsManager)
        check_integrity(log_settings, dict)
        check_integrity(is_monitoring_enabled, bool)
        
        # everything is fine

        # 1. init steering command
        # includes param setup, buffer creation
        # NOTE init is system action and so is done implicitly with the hub
        # initialization
        interscalehub_adapter = InterscaleHubAdapter(
            direction,
            configurations_manager,
            log_settings,
            is_monitoring_enabled,
            sci_params_xml_path_filename=p_sci_params_xml_path_filename)
        interscalehub_adapter.execute_init_command()

        # Get steering command in rank0 and share this with the other ranks
        if interscalehub_adapter.rank == 0:
            user_action_command = input()
        else:
            user_action_command = None
        user_action_command = interscalehub_adapter.comm.bcast(
            user_action_command, root=0)
        # print(f"rank: {rank}, user_action_command:{user_action_command}")

        # NOTE Application Manager sends the control commands with parameters in
        # the following specific format as a string via stdio:
        # {'STEERING_COMMAND': {'<Enum SteeringCommands>': <Enum value>}, 'PARAMETERS': <value>}
        
        # For example:
        # {'STEERING_COMMAND': {'SteeringCommands.START': 2}, 'PARAMETERS': 1.2}        

        # convert the received string to dictionary
        control_command = ast.literal_eval(user_action_command.strip())
        # get steering command
        steering_command_dictionary = control_command.get(COMMANDS.STEERING_COMMAND.name)
        current_steering_command = next(iter(steering_command_dictionary.values()))

        # 2. execute if steering command is 'START'
        if current_steering_command == SteeringCommands.START:
            # fetch id_first_spike_detector
            spike_detectors_ids = control_command.get(COMMANDS.PARAMETERS.name)[1]
            # print(f'current_steering_command: {current_steering_command} '
            #       f'id_first_spike_detector: {id_first_spike_detector[0]}')
            # execute the START command
            # receive, pivot, transform, send
            interscalehub_adapter.execute_start_command(spike_detectors_ids)
            
            # execute the END command
            interscalehub_adapter.execute_end_command()
            
            # exit with success code
            sys.exit(0)
        else:
            print(f'unknown command: {current_steering_command}', file=sys.stderr)
            sys.exit(1)

    else:
        print(f'missing argument[s]; required: 6, received: {len(sys.argv)}')
        print(f'Argument list received: {str(sys.argv)}')
        sys.exit(1)
