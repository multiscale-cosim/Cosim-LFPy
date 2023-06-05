# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
import os

from EBRAINS_RichEndpoint.application_companion.resource_usage_monitor import ResourceUsageMonitor
from EBRAINS_RichEndpoint.application_companion.common_enums import Response
from EBRAINS_RichEndpoint.application_companion.db_manager_file import DBManagerFile
from EBRAINS_RichEndpoint.application_companion.affinity_manager import AffinityManager
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class ResourceMonitorAdapter:
    def __init__(self, configurations_manager, log_settings,
                 action_pid,
                 action_process_name,
                 poll_interval=1.0,  # default is 1 second,
                 ):
         
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
                                        name="Resource Uagae Adapter",
                                        log_configurations=self._log_settings)
        
        self.__action_pid = action_pid
        self.__action_process_name = action_process_name
        self.__poll_interval = poll_interval
        self.__affinity_manager = AffinityManager(self._log_settings,
                                                   self._configurations_manager)
        # get affinity mask of the action
        self.__action_bound_with_cores = self.__get_action_affinity()
        self.__logger.debug("logger is configured.")

    def __get_action_affinity(self):
         
         affinity_mask = self.__affinity_manager.get_affinity(self.__action_pid)
         self.__logger.debug(f"{self.__action_process_name} "
                             f"<{self.__action_pid}> "
                             f"affinity mask: <{affinity_mask}>")
         return affinity_mask
    
    def start_monitoring(self):
            self.__logger.info("start monitoring for action: "
                               f"<{self.__action_process_name}> "
                               f"pid{self.__action_pid} ")
            
            # initialize resource usage monitor
            self.__resource_usage_monitor = ResourceUsageMonitor(
                self._log_settings,
                self._configurations_manager,
                self.__action_pid,
                self.__action_bound_with_cores,
                self.__action_process_name,
                self.__poll_interval
                )
            # start monitoring
            # Case a, monitoring could not be started
            if self.__resource_usage_monitor.start_monitoring() == Response.ERROR:
                self.__logger.error(
                        f'Could not start monitoring for <NEST>: {self.__action_pid}')
                return Response.ERROR
            
            # Case b: monitoring starts
            self.__logger.debug("started monitoring the resource usage.")
            return Response.OK
        
    def stop_monitoring(self):
        self.__resource_usage_monitor.keep_monitoring = False
        resource_usage_summary =\
                self.__resource_usage_monitor.get_resource_usage_stats(0)
        self.__logger.debug(f"Resource Usage stats: "
                            f"{resource_usage_summary.items()}")
        # get directory to save the resource usage statistics
        try:
            metrics_output_directory = \
                self._configurations_manager.get_directory(
                    DefaultDirectories.MONITORING_DATA)
            # exception raised, if default directory does not exist
        except KeyError:
            # create a new directory
            metrics_output_directory = \
                self._configurations_manager.make_directory(
                    'Resource usage metrics', directory_path='AC results')
        
        # path to JSON file for dumping the monitoring data
        metrics_file = os.path.join(metrics_output_directory,
                                    f'{self.__action_process_name}_'
                                    f'pid_{self.__action_pid}'
                                    '_resource_usage_metrics.json')
        # dump the monitoring data
        # NOTE monitoring data is dumped in to JSON files
        # initialize JSON files handler
        self.__db_manager_file = DBManagerFile(
            self._log_settings, self._configurations_manager)
        self.__db_manager_file.write(metrics_file,
                                     resource_usage_summary)
        return Response.OK