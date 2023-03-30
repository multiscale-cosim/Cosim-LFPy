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


##############################################################################
'''
Protjans 2014 simulaiton model with NEST
For inspiration, look
run_microcircuit.py (https://github.com/mfahdaz/nest-simulator/blob/ecc373fe83a82d133f562a4ed6cf180d07c39578/pynest/examples/Potjans_2014/run_microcircuit.py)
'''
##############################################################################
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
        self.__sci_params = Xml2ClassParser(sci_params_xml_path_filename, self.__logger)
        self.__parameters = Parameters(self.__path_to_parameters_file)

        # TODO 3. create MPI intercomm

        # 4. Initialize port_names in the format as per nest-simulator
        # NOTE The MPI port_name needs to be in string format and must be sent to
        # nest-simulator in the following pattern:
        # "endpoint_address":<port name>
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
            self.__logger.debug(f"running interscalehub: {interscalehub}")
            # NEST_TO_LFPy RECEIVER endpoint
            if interscalehub.get(
                    INTERSCALE_HUB.DATA_EXCHANGE_DIRECTION.name) ==\
                    DATA_EXCHANGE_DIRECTION.NEST_TO_LFPy.name:
                # get mpi port name
                self.__interscalehub_NEST_TO_LFPy_address =\
                    "endpoint_address:"+interscalehub.get(
                        INTERSCALE_HUB.MPI_CONNECTION_INFO.name)
                self.__logger.debug("Interscalehub_nest_to_tvb_address: "
                                    f"{self.__interscalehub_NEST_TO_LFPy_address}")

    def execute_init_command(self):
        self.__logger.debug("executing INIT command")
        # 1. configure simulation model
        self.__logger.info("configure the network")
        # TODO configure simulator
        self.__logger.info("preparing the simulator, and "
                           "establishing the connections")
        # setup connections with InterscaleHub
        self.__logger.info("connections are made")
        self.__logger.debug("INIT command is executed")
        # 2. return local minimum step size
        return local_minimum_step_size

    def execute_start_command(self, global_minimum_step_size):
        self.__logger.debug("executing START command")
        self.__logger.debug(f'global_minimum_step_size: {global_minimum_step_size}')
        count = 0.0
        self.__logger.debug('starting simulation')
        while count * global_minimum_step_size < self.__parameters.simulation_time:
            self.__logger.info(f"simulation run counter: {count+1}")
            # TODO run simulation
            count += 1

        self.__logger.info('ARBOR simulation is finished')
        self.__logger.info("cleaning up ARBOR")
        nest.Cleanup()
        # self.execute_end_command()

    def execute_end_command(self):
        self.__logger.debug("executing END command")
        # post processing
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