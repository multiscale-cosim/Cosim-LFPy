# ----------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ----------------------------------------------------------------------------
import numpy as np
import time
import os
import sys
import pickle
import base64
import ast

from science.models.Potjans import network
from science.parameters.Potjans.stimulus_params import stim_dict
from science.parameters.Potjans.network_params import net_dict
from science.parameters.Potjans.sim_params import sim_dict
# import nest

from common.utils.security_utils import check_integrity

from EBRAINS_RichEndpoint.application_companion.common_enums import SteeringCommands, COMMANDS, INTERCOMM_TYPE
from EBRAINS_RichEndpoint.application_companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR
from EBRAINS_RichEndpoint.application_companion.common_enums import INTEGRATED_INTERSCALEHUB_APPLICATION as INTERSCALE_HUB
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.configurations_manager import ConfigurationsManager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers.xml2class_parser import Xml2ClassParser
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_EXCHANGE_DIRECTION 

'''
Potjans 2014 simulaiton model with NEST
based on example: https://github.com/mfahdaz/nest-simulator/blob/ecc373fe83a82d133f562a4ed6cf180d07c39578/pynest/examples/Potjans_2014/
'''

class NestAdapter:
    def __init__(self, p_configurations_manager, p_log_settings,
                 p_interscalehub_addresses,
                 sci_params_xml_path_filename=None):
        # 1. set up logger
        self._log_settings = p_log_settings
        self._configurations_manager = p_configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
        name="nest_adapter",
        log_configurations=self._log_settings,
        target_directory=DefaultDirectories.SIMULATION_RESULTS)
        
        # 2. Load parameters for the model
        # TODO load parameters from a single source
        self.__path_to_parameters_file = self._configurations_manager.get_directory(
        directory=DefaultDirectories.SIMULATION_RESULTS)
        # self.__sci_params = Xml2ClassParser(sci_params_xml_path_filename, self.__logger)
        # self.__parameters = Parameters(self.__path_to_parameters_file)

        # 4. Initialize port_names in the format as per nest-simulator
        # NOTE The MPI port_name needs to be in string format and must be sent to
        # nest-simulator in the following pattern:
        # "endpoint_address":<port name>
        self.__interscalehub_NEST_TO_LFPy_address = None
        self.__init_port_names(p_interscalehub_addresses)
        self.__simulator = None
        self.__logger.debug(f"running on host_name:{os.uname()}")
        self.__logger.info("initialized")

    def __init_port_names(self, interscalehub_addresses):
        '''
        helper function to prepare the port_names in the following format:
        "endpoint_address":<port name>
        '''
        # TODO refactor to match the (bi-directional) interscaleHub
        for interscalehub in interscalehub_addresses:
            # endpoint to receive data from simulator
            if interscalehub.get(
                    INTERSCALE_HUB.DATA_EXCHANGE_DIRECTION.name) ==\
                    DATA_EXCHANGE_DIRECTION.NEST_TO_LFPY.name and interscalehub.get(
                    INTERSCALE_HUB.INTERCOMM_TYPE.name) == INTERCOMM_TYPE.RECEIVER.name:
                # get mpi port name
                self.__interscalehub_NEST_TO_LFPy_address =\
                    "endpoint_address:"+interscalehub.get(
                        INTERSCALE_HUB.MPI_CONNECTION_INFO.name)
                self.__logger.debug("Interscalehub_receive_from_simulator_address: "
                                f"{self.__interscalehub_NEST_TO_LFPy_address}")

    def execute_init_command(self):
        self.__logger.debug("executing INIT command")
        # 1. configure simulation model
        self.__logger.info("configure the network")
        self.__simulator = network.Network(sim_dict, net_dict, stim_dict)
        # setup connections with InterscaleHub
        self.__logger.info("preparing the simulator, and "
                           "establishing the connections")
        self.__simulator.create(self.__interscalehub_NEST_TO_LFPy_address)
        self.__logger.debug(f"connecting simulator")
        self.__simulator.connect()
        self.__logger.info("connections are made")
        self.__logger.debug("INIT command is executed")
        # 2. return local minimum step size
        # TODO determine the local minimum step size
        local_minimum_step_size = 1.5  # NOTE hardcoded, will change later
        return local_minimum_step_size

    def execute_start_command(self, global_minimum_step_size):
        self.__logger.debug("executing START command")
        self.__logger.debug(f'global_minimum_step_size: {global_minimum_step_size}')
        self.__logger.debug('starting simulation')
        # NOTE following is relavent if it is a cosimulation
        count = 0.0
        min_step_size = 1.2
        while count * min_step_size < sim_dict['t_sim']:
            self.__logger.info(f"simulation run counter: {count+1}")
            self.__logger.info(f"total simulation time: {sim_dict['t_sim']}")
            # TODO run simulation
            self.__simulator.simulate(min_step_size)
            count += 1
        # NOTE two subsequent calls to simulate ?
        # self.__simulator.simulate(sim_dict['t_presim'])
        # self.__simulator.simulate(sim_dict['t_sim'])

        
        self.__logger.info('NEST simulation is finished')
        self.__logger.info("cleaning up NEST")
        self.__simulator.cleanup()
        # nest.Cleanup()
        # self.execute_end_command()

    def execute_end_command(self):
        self.__logger.debug("executing END command")
        # post processing
        # Plot a spike raster of the simulated neurons and a box plot of the firing
        # rates for each population.
        # For visual purposes only, spikes 100 ms before and 100 ms after the thalamic
        # stimulus time are plotted here by default.
        # The computation of spike rates discards the presimulation time to exclude
        # initialization artifacts.

        raster_plot_interval = np.array([stim_dict['th_start'] - 100.0,
                                        stim_dict['th_start'] + 100.0])
        firing_rates_interval = np.array([sim_dict['t_presim'],
                                        sim_dict['t_presim'] + sim_dict['t_sim']])
        self.__simulator.evaluate(raster_plot_interval, firing_rates_interval)
        self.__logger.debug("post processing is done")


if __name__ == "__main__":
    if len(sys.argv) == 5:        
        # 1. parse arguments
        # unpickle configurations_manager object
        configurations_manager = pickle.loads(base64.b64decode(sys.argv[1]))
        # unpickle log_settings
        log_settings = pickle.loads(base64.b64decode(sys.argv[2]))
        # get science parameters XML file path
        p_sci_params_xml_path_filename = sys.argv[3]
        # get interscalehub connection details
        p_interscalehub_address = pickle.loads(base64.b64decode(sys.argv[4]))

        # 2. security check of pickled objects
        # it raises an exception, if the integrity is compromised
        check_integrity(configurations_manager, ConfigurationsManager)
        check_integrity(log_settings, dict)
        check_integrity(p_interscalehub_address, list)

        # 3. everything is fine, configure simulator
        nest_adapter = NestAdapter(
            configurations_manager,
            log_settings,
            p_interscalehub_address,
            sci_params_xml_path_filename=p_sci_params_xml_path_filename)

        # 4. execute 'INIT' command which is implicit with when laucnhed
        local_minimum_step_size = nest_adapter.execute_init_command()

        # 5. send the pid and the local minimum step size to Application Manager
        # as a response to 'INIT' as per protocol
        
        # NOTE Application Manager expects a string in the following format:
        # {'PID': <pid>, 'LOCAL_MINIMUM_STEP_SIZE': <step size>}

        # prepare the response
        pid_and_local_minimum_step_size = \
            {SIMULATOR.PID.name: os.getpid(),
            SIMULATOR.LOCAL_MINIMUM_STEP_SIZE.name: local_minimum_step_size}
        
        # send the response
        # NOTE Application Manager will read the stdout stream via PIPE
        print(f'{pid_and_local_minimum_step_size}')

        # 6. fetch next command from Application Manager
        user_action_command = input()

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
        
        # 7. execute if steering command is 'START'
        if current_steering_command == SteeringCommands.START:
            # fetch global minimum step size
            global_minimum_step_size = control_command.get(COMMANDS.PARAMETERS.name)
            # execute the command
            nest_adapter.execute_start_command(global_minimum_step_size)
            nest_adapter.execute_end_command()
            # exit with success code
            sys.exit(0)
        else:
            print(f'unknown command: {current_steering_command}', file=sys.stderr)
            sys.exit(1)
    else:
        print(f'missing argument[s]; required: 5, received: {len(sys.argv)}')
        print(f'Argument list received: {str(sys.argv)}')
        sys.exit(1)