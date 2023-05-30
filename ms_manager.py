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
#       Team: Multi-scale Simulation and Design
#
# ------------------------------------------------------------------------------
import os
import json

# Co-Simulator imports
from common import args
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import enums
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import variables
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import comm_settings_xml_manager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import services_deployment_xml_manager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import plan_xml_manager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import xml_tags
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers.variables import CO_SIM_EXECUTION_ENVIRONMENT
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import variables_manager
# from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import parameters_xml_manager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import actions_xml_manager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import arranger
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers import configurations_manager
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_Launcher.launching_manager import LaunchingManager


class MSManager:
    """
        Class representing the Co-Simulator tool
    Methods:
    --------
        run(args=None)
            Entry point to the Co-Simulator and executes the main loop of the tool
    """

    def __init__(self):
        # general members
        self.__args = None
        self.__arranger = None
        self.__configurations_manager = None
        # self.__logs_root_dir = None
        self.__logger = None
        # self.__launcher = None

        # Environment
        self.__variables_manager = None

        # XML configuration files managers
        self.__comm_settings_xml_manager = None
        self.__actions_xml_manager = None
        # self.__parameters_xml_manager = None
        self.__plan_xml_manager = None
        self.__services_deployment_xml_manager = None

        # dictionaries
        self.__communication_settings_dict = {}
        self.__action_plan_parameters_dict = {}
        self.__action_plan_variables_dict = {}
        self.__action_plan_dict = {}
        self.__items_to_be_arranged = {}

        self.__actions_popen_args_dict = {}
        self.__actions_sci_params_xml_files_dict = {}

        # self.__parameters_parameters_dict = {}
        self.__parameters_parameters_for_json_file_dict = {}
        # self.__parameters_variables_dict = {}
        self.__services_deployment_dict = {}

        # Communication Settings (zmq ports) used by Co-Sim Components
        self.__co_sim_comm_settings_xml_file = ''  # empty, reference to is gotten from action plan XML file

        # XML referring to the HPC deployment of the Co-Sim Components
        self.__co_sim_services_deployment_xml_file = ''  # empty, reference to is gotten from action plan XML file

        # logging settings
        self.__logger_settings = {}

    def generate_parameters_json_file(self):
        """
            Dumps into the /path/to/co_sim/results/dir/filename.json file
            the parameters gathered from the parameters XML file
        :return:
            JSON_FILE_ERROR: reporting error during the parameter JSON file
            OK: parameter JSON file was generated properly
        """
        # TODO: exception management when the file cannot be created

        results_dir = self.__configurations_manager.get_directory(DefaultDirectories.RESULTS)
        json_output_filename = \
            self.__parameters_parameters_for_json_file_dict[xml_tags.CO_SIM_XML_CO_SIM_PARAMS_FILENAME]
        json_output_path_filename = os.path.join(results_dir, json_output_filename)

        try:
            with open(json_output_path_filename, 'w') as json_output_file:
                json.dump(
                    self.__parameters_parameters_for_json_file_dict[xml_tags.CO_SIM_XML_CO_SIM_PARAMS_JSON_FILE],
                    json_output_file)
        except OSError:
            self.__logger.error('{} cannot be created, OS error'.format(json_output_path_filename))
            return enums.CoSimulatorReturnCodes.JSON_FILE_OS_ERROR

        self.__logger.info('Co-Simulation parameters were transformed successfully')
        self.__logger.info('Co-Simulation parameters: {}'.format(json_output_path_filename))
        return enums.CoSimulatorReturnCodes.OK

    def run(self):
        """
            Entry point of the Co-Simulation Co-Simulator tool
        :param:
            There is no parameters to be used since argparse takes the sys.argv by default
        :return:
            common.enums.CoSimulatorReturnCodes
        """
        ########
        # STEP 1 - Checking command line parameters
        ########
        try:
            # self.__args = args.arg_parse()
            self.__args = args.get_parsed_CLI_arguments()
        except SystemExit:
            # argument parser has reported some issue with the arguments
            return enums.CoSimulatorReturnCodes.PARAMETER_ERROR
        
        # set whether interactive steering is enabled from CLI arguments
        self.__is_interactive = self.__args.interactive

        ########
        # STEP 2 - Setting Up the Configuration Manager
        ########

        ####################
        # instantiate configuration manager
        self.__configurations_manager = configurations_manager.ConfigurationsManager()

        # get path to set up the output directories
        default_dir = self.__configurations_manager.get_configuration_settings(
            'output_directory', self.__args.global_settings)

        # setup default directories (Output, Output/Results, Output/Logs,
        # Output/Figures, Output/Monitoring_DATA)
        self.__configurations_manager.setup_default_directories(default_dir['output_directory'])

        # load common settings for the logging
        self.__logger_settings = self.__configurations_manager.get_configuration_settings(
            'log_configurations', self.__args.global_settings)

        self.__logger = self.__configurations_manager.load_log_configurations(
            name=__name__, log_configurations=self.__logger_settings)
        self.__logger.info('Co-Simulator STEP 1 done, args are parsed.')
        self.__logger.info('Co-Simulator STEP 2 done, output directories are setup.')

        ########
        # STEP 3 - Setting Up CO_SIM_* Variables by means of the Variables Manager
        ########
        self.__logger.info('Co-Simulator STEP 3 running')
        self.__variables_manager = \
            variables_manager.VariablesManager(self.__logger_settings, self.__configurations_manager)

        # STEP 3.1 - Setting Up the output location (path) for results
        # TODO handle case when set value() fails
        self.__variables_manager.set_value(
            variables.CO_SIM_RESULTS_PATH,
            self.__configurations_manager.get_directory(DefaultDirectories.OUTPUT)
        )

        self.__logger.info(
            f'Co-Simulator STEP 3 done, Co-Simulation results location: '
            f'{self.__variables_manager.get_value(variables.CO_SIM_RESULTS_PATH)}')

        ########
        # STEP 4 - Co-Simulation Plan
        ########
        self.__logger.info('Co-Simulator STEP 4, dissecting Co-Simulation Action Plan')
        self.__plan_xml_manager = \
            plan_xml_manager.PlanXmlManager(
                log_settings=self.__logger_settings,
                configurations_manager=self.__configurations_manager,
                xml_filename=self.__args.action_plan,
                name='PlanXmlManager')

        # STEP 4.1 - Dissecting the Co-Simulation Plan XML file
        # NOTE: <variables> section could/can contain references to Environment Variables,
        #       e.g. ${HOME}, ${CO_SIM_ROOT_PATH}
        #       In this point, the Environment Variables references will be replaced with their actual values
        if not self.__plan_xml_manager.dissect() == enums.XmlManagerReturnCodes.XML_OK:
            return enums.CoSimulatorReturnCodes.XML_ERROR

        # STEP 4.2 - Getting the variables found on the Co-Simulation Plan XML file
        #
        self.__action_plan_variables_dict = self.__plan_xml_manager.get_variables_dict()

        # STEP 4.3 -    Validating the references to the CO_SIM_* variables
        #               by filling up the environment variables dictionary
        if not enums.VariablesReturnCodes.VARIABLE_OK == \
               self.__variables_manager.set_co_sim_variable_values_from_variables_dict(
                   self.__action_plan_variables_dict):
            return enums.CoSimulatorReturnCodes.VARIABLE_ERROR

        # Parameters -> Could contain references to CO_SIM_ variables and become new CO_SIM_ variables
        # STEP 4.4 - Getting the parameters found on the Co-Simulation Plan XML file
        self.__action_plan_parameters_dict = self.__plan_xml_manager.get_parameters_dict()
        
        # STEP 4.5 -    Validating the references to the CO_SIM_* variables on the <parameters> sections
        #               by creating the new CO_SIM_* variables by means of the variables manager
        if not enums.ParametersReturnCodes.PARAMETER_OK == \
               self.__variables_manager.create_variables_from_parameters_dict(self.__action_plan_parameters_dict):
            return enums.CoSimulatorReturnCodes.PARAMETER_ERROR
        
        # STEP 4.6 - Creates Co-Simulation variables based on the information
        #            set on the variables and parameters sections of the processing XML action plan file
        #            e.g. CO_SIM_EXECUTION_ENVIRONMENT = <local|cluster>
        if not enums.VariablesReturnCodes.VARIABLE_OK == \
               self.__variables_manager.create_co_sim_run_time_variables():
            return enums.CoSimulatorReturnCodes.VARIABLE_ERROR

        # Action Plan -> ordered and grouped sequence of actions to achieve the Co-Simulation Experiment
        # STEP 4.7 - Getting the action plan per se
        self.__action_plan_dict = self.__plan_xml_manager.get_action_plan_dict()

        self.__logger.info('{} -> {}'.format(variables.CO_SIM_ROOT_PATH,
                                             self.__variables_manager.get_value(variables.CO_SIM_ROOT_PATH)))
        self.__logger.info('{} -> {}'.format(variables.CO_SIM_ACTIONS_PATH,
                                             self.__variables_manager.get_value(variables.CO_SIM_ACTIONS_PATH)))
        self.__logger.info('{} -> {}'.format(variables.CO_SIM_ROUTINES_PATH,
                                             self.__variables_manager.get_value(variables.CO_SIM_ROUTINES_PATH)))
        self.__logger.info('{} -> {}'.format(variables.CO_SIM_COMMUNICATION_SETTINGS_PATH,
                                             self.__variables_manager.get_value(
                                                 variables.CO_SIM_COMMUNICATION_SETTINGS_PATH)))

        self.__logger.info('Co-Simulator STEP 4 done')

        ########
        # STEP 5 - Processing Co-Simulation Parameters
        ########

        # self.__logger.info('Co-Simulator STEP 5, dissecting Co-Simulation parameters')
        # self.__parameters_xml_manager = \
        #     parameters_xml_manager.ParametersXmlManager(configuration_manager=self.__configuration_manager,
        #                                                        logger=self.__logger,
        #                                                        xml_filename=self.__args.parameters)

        # # STEP 5.1 - Dissecting Co-Simulation Parameters XML file
        # if not self.__parameters_xml_manager.dissect() == enums.XmlManagerReturnCodes.XML_OK:
        #     return enums.CoSimulatorReturnCodes.XML_ERROR

        # # STEP 5.2 - Getting the variables found in the Co-Simulation Parameters file
        # self.__parameters_variables_dict = self.__parameters_xml_manager.get_variables_dict()

        # # STEP 5.3 - Getting the parameters found in the Co-Simulation Parameters file
        # self.__parameters_parameters_dict = self.__parameters_xml_manager.get_parameters_dict()

        # # STEP 5.4 - Getting the Co-Simulation parameters to be dumped into a json file
        # self.__parameters_parameters_for_json_file_dict = self.__parameters_xml_manager.get_parameter_for_json_dict()

        # self.__logger.info('Co-Simulation parameters loaded from {}'.format(self.__args.parameters))
        # self.__logger.info('Co-Simulator STEP 5 done')

        ########
        # STEP 5 - Co-Simulation Components Settings
        ########
        self.__logger.info('Co-Simulator STEP 5, Co-Simulation Components Settings')

        # # STEP 5.1 - Dissecting the Co-Simulation Communication Settings XML file
        self.__logger.info('Co-Simulator STEP 5.1, dissecting Co-Simulation Communication Settings XML file')
        self.__co_sim_comm_settings_xml_file = \
            self.__variables_manager.get_value(variables.CO_SIM_COMMUNICATION_SETTINGS_XML)
        self.__logger.info('{} -> {}'.format(variables.CO_SIM_COMMUNICATION_SETTINGS_XML,
                                             self.__variables_manager.get_value(
                                                 variables.CO_SIM_COMMUNICATION_SETTINGS_XML)))

        self.__comm_settings_xml_manager = \
            comm_settings_xml_manager.CommunicationSettingsXmlManager(log_settings=self.__logger_settings,
                                                                      configurations_manager=self.__configurations_manager,
                                                                      xml_filename=self.__co_sim_comm_settings_xml_file,
                                                                      name="CommunicationSettingsXmlManager")

        if not self.__comm_settings_xml_manager.dissect() == enums.XmlManagerReturnCodes.XML_OK:
            return enums.CoSimulatorReturnCodes.XML_ERROR

        self.__communication_settings_dict = self.__comm_settings_xml_manager.get_communication_settings_dict()

        if self.__action_plan_variables_dict[CO_SIM_EXECUTION_ENVIRONMENT].upper() != "LOCAL":
            self.__logger.info('Co-Simulator STEP 5.2, Using HPC Mode')
            self.__logger.info('Co-Simulator STEP 5.2, dissecting Co-Simulation Services Deployment XML file')
            self.__co_sim_services_deployment_xml_file = \
                self.__variables_manager.get_value(variables.CO_SIM_SERVICES_DEPLOYMENT_XML)
            self.__logger.info('{} -> {}'.format(variables.CO_SIM_SERVICES_DEPLOYMENT_XML,
                                                 self.__variables_manager.get_value(
                                                     variables.CO_SIM_SERVICES_DEPLOYMENT_XML)))

            self.__services_deployment_xml_manager = \
                services_deployment_xml_manager.ServicesDeploymentXmlManager(
                    log_settings=self.__logger_settings,
                    configurations_manager=self.__configurations_manager,
                    variables_manager=self.__variables_manager,
                    xml_filename=self.__co_sim_services_deployment_xml_file,
                    name="ServicesDeploymentXmlManager")

            if not self.__services_deployment_xml_manager.dissect() == enums.XmlManagerReturnCodes.XML_OK:
                return enums.CoSimulatorReturnCodes.XML_ERROR

            self.__services_deployment_dict = self.__services_deployment_xml_manager.get_services_deployment_dict()

        self.__logger.info('Co-Simulator STEP 5 done')

        ########
        # STEP 6 - Co-Simulation Actions (processing the XML configuration files)
        ########
        self.__logger.info('Co-Simulator STEP 6, dissecting Co-Simulation Actions XML files')
        # STEP 6.1 - Getting the Actions Popen arguments, the CO_SIM_ variables transformation is performed
        self.__actions_xml_manager = actions_xml_manager.ActionsXmlManager(
            self.__logger_settings,
            self.__configurations_manager,
            self.__variables_manager,
            self.__action_plan_dict
        )

        if not self.__actions_xml_manager.dissect() == enums.XmlManagerReturnCodes.XML_OK:
            return enums.CoSimulatorReturnCodes.XML_ERROR

        self.__actions_popen_args_dict = self.__actions_xml_manager.get_actions_popen_arguments_dict()
        self.__actions_sci_params_xml_files_dict = self.__actions_xml_manager.get_actions_sci_params_xml_files_dict()

        self.__logger.info('Co-Simulator STEP 6 done')

        ########
        # STEP 7 - Arranging run time environment
        ########
        self.__logger.info('Co-Simulator STEP 7, arranging environment')
        self.__items_to_be_arranged = self.__plan_xml_manager.get_items_to_be_arranged_dict()

        self.__arranger = arranger.Arranger(
            self.__logger_settings,
            self.__configurations_manager,
            self.__variables_manager,
            self.__items_to_be_arranged
        )

        if not self.__arranger.arrange() == enums.ArrangerReturnCodes.OK:
            return enums.CoSimulatorReturnCodes.ARRANGER_ERROR
        self.__logger.info('Co-Simulator STEP 7 done')

        ########
        # STEP 8 - Converting Co-Simulation parameters from XML into JSON
        ########
        # self.__logger.info('Co-Simulator STEP 8, transforming Co-Simulation parameters')
        # if not self.generate_parameters_json_file() == enums.CoSimulatorReturnCodes.OK:
        #     return enums.CoSimulatorReturnCodes.JSON_FILE_ERROR
        # self.__logger.info('Co-Simulator STEP 8 done')

        ########
        # STEP 9 - Launching the Action Plan
        ########
        self.__logger.info('Co-Simulator STEP 9, carrying out the Co-Simulation Action Plan Strategy')
        launching_manager = LaunchingManager(action_plan_dict=self.__action_plan_dict,  # actions
                                             action_plan_variables_dict=self.__action_plan_variables_dict,
                                             # <local|cluster>
                                             action_plan_parameters_dict=self.__action_plan_parameters_dict,  # paths
                                             actions_popen_args_dict=self.__actions_popen_args_dict,
                                             # mpirun/srun parameters
                                             log_settings=self.__logger_settings,  # logging configurations
                                             configurations_manager=self.__configurations_manager,  # config manager
                                             # scientific parameters
                                             actions_sci_params_dict=self.__actions_sci_params_xml_files_dict,
                                             # if interactive steering is enabled
                                             is_interactive=self.__is_interactive,
                                             # zmq ports
                                             communication_settings_dict=self.__communication_settings_dict,
                                             # nodes where to deploy Co-Sim services
                                             services_deployment_dict=self.__services_deployment_dict
                                             )

        if not launching_manager.carry_out_action_plan() == enums.LauncherReturnCodes.LAUNCHER_OK:
            self.__logger.error('Error(s) were reported, check the errors log on {}'.format(
                self.__variables_manager.get_value(variables.CO_SIM_RESULTS_PATH)))
            return enums.CoSimulatorReturnCodes.LAUNCHER_ERROR
        # if not self.__launcher.carry_out_action_plan() == common.enums.LauncherReturnCodes.LAUNCHER_OK:
        #     self.__logger.error('Error(s) were reported, check the errors log on {}'.format(
        #         self.__variables_manager.get_value(common.variables.CO_SIM_RESULTS_PATH)))
        #     return common.enums.CoSimulatorReturnCodes.LAUNCHER_ERROR
        self.__logger.info('Co-Simulator STEP 8 done')

        ########
        # STEP 10 - Finishing
        ########
        self.__logger.info('Information about Co-Simulation process could be found on: {}'.format(
            self.__variables_manager.get_value(variables.CO_SIM_RESULTS_PATH)))
        self.__logger.info('END: Co-Simulation Co-Simulator')

        return enums.CoSimulatorReturnCodes.OK