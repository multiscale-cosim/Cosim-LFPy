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
import os
import re

from EBRAINS_RichEndpoint.application_companion.common_enums import SERVICE_COMPONENT_CATEGORY
from EBRAINS_RichEndpoint.application_companion.common_enums import Response
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import variables
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers import constants


# TODO setup XML files for the settings
# TODO srun command will then be prepared by Launching Manager
def deployment_command(logger, is_execution_environment_hpc, service,
                       default_cosim_nodelist_for_service,
                       target_nodelist_from_xml,
                       *args):
    """
    helper function to get the command to deploy the service locally or
    on HPC systems.
    """
    # Case a, service is to be deployed on compute nodes (HPC)
    # then, prepare srun command
    if is_execution_environment_hpc:
        return command_to_deploy_on_hpc_systems(logger,
                                                args,
                                                service,
                                                target_nodelist_from_xml,
                                                default_cosim_nodelist_for_service)

    # Case b, service is to be deployed locally
    else:
        return command_to_deploy_on_local_system(logger, args, service)


def command_to_deploy_on_hpc_systems(logger, args, service,
                                     target_nodelist_from_xml,
                                     default_cosim_nodelist_for_service,):
    command = []
    if "--nodelist" not in target_nodelist_from_xml:
        # nodelist = deployment_settings[service_component_name]
        # __original__ target_nodelist = cosim_slurm_nodes_mapping(logger)[default_cosim_nodelist_for_service]
        target_nodelist = default_cosim_nodelist_for_service

        command.append(f"--nodelist={target_nodelist}")
        logger.debug(f"target nodelist={target_nodelist}")
    else:
        command.append(target_nodelist_from_xml)
    # command with target nodelist
    logger.debug(f"running command:{command}")
    # append the service specific arguments
    command.extend(["python3", f"{service}"])
    for arg in args[0]:
        command.append(arg)
    srun_command_with_args = default_srun_command + command
    logger.debug(f"srun command with arguments:{srun_command_with_args}")
    return srun_command_with_args


def command_to_deploy_on_local_system(logger, args, service):
    command = []
    logger.debug(f"preparing command for {service} to deploy locally")
    # append the service arguments required to instantiate and run it
    command.append("python3")
    command.append(service)
    for arg in args[0]:
        command.append(arg)
    logger.debug(f"command with arguments:{command}")
    return command


#######################
# default srun command
#######################
default_srun_command = ["srun",
                        "--exact",
                        "--label",
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=1",
                        "--cpu-bind=none",
                        "--gres=gpus:0"
                        ]


###########################################
# settings to deploy on given compute node
###########################################

# NOTE the variable CO_SIM_SLURM_NODE_xxx is translated to the allocated
# compute resources such that e.g. if two compute nodes are allocated namely
# xxx056 and xxx057 then CO_SIM_SLURM_NODE_000 is translated to compute node
# xxx056 and similarly CO_SIM_SLURM_NODE_001 is translated to compute node xxx057
deployment_settings = {
    SERVICE_COMPONENT_CATEGORY.APPLICATION_COMPANION.name: "CO_SIM_SLURM_NODE_000",
    SERVICE_COMPONENT_CATEGORY.COMMAND_AND_CONTROL.name: "CO_SIM_SLURM_NODE_000",
    SERVICE_COMPONENT_CATEGORY.STEERING_SERVICE.name: "CO_SIM_SLURM_NODE_000",
    SERVICE_COMPONENT_CATEGORY.PROXY_MANAGER_SERVER.name: "CO_SIM_SLURM_NODE_000",
    SERVICE_COMPONENT_CATEGORY.ORCHESTRATOR.name: "CO_SIM_SLURM_NODE_000"
    }


def is_salloc(logger, n_nodes, node_range_length):
    """checks if salloc is already successful"""
    if node_range_length == 0:
        logger.error('SLURM_NODELIST environment variable has not '
                     'been set yet, use "salloc"')
        return Response.ERROR
    elif node_range_length == 1 and node_range_length != n_nodes:
        # There is no match between SLURM_NNODES and SLURM_NODELIST
        logger.error('SLURM_NODELIST does not match with '
                     'SLURM_NODELIST, it might be "salloc" failed')
        return Response.ERROR

    # otherwise, all is well and salloc is successful
    return Response.OK


# NOTE this function is adapted from
# EBRAINS_ConfigManager/...variables_manager.py -> __creates_co_sim_vars_from_slurm_env_vars()
def cosim_slurm_nodes_mapping(logger):
    """
        Populates the CO_SIM_* variables from the SLURM_* environment variables
        IMPORTANT: It is assumed that salloc has been executed and the resources
                    haven assigned properly (end-user task)
    :return
        node id
    """
    cosim_slurm_nodes = {}
    # No. of requested HPC nodes
    try:
        n_nodes = int(os.environ['SLURM_NNODES'])
        cosim_slurm_nodes[variables.CO_SIM_SLURM_NNODES] = \
            {constants.CO_SIM_VARIABLE_DESCRIPTION: 'SLURM_NNODES',
                constants.CO_SIM_VARIABLE_VALUE: n_nodes}
    except KeyError:
        logger.exception('SLURM_NNODES environment variable has not been set yet, use "salloc"')
        return Response.ERROR

    # Since SLURM_NNODES is set, meaning SLURM_NODELIST must be set as well,
    # e.g. SLURM_NODELIST=jsfc056       -> 1 Node
    #      SLURM_NODELIST=jsfc[056-057] -> 2 Nodes
    slurm_node_list_prefix_and_range = re.split('\[|\]', os.environ['SLURM_NODELIST'])
    node_range_length = len(slurm_node_list_prefix_and_range)

    # check if salloc is successful
    if is_salloc(logger, n_nodes, node_range_length) == Response.ERROR:
        # Case a: salloc is not successful
        return Response.ERROR

    # Case b: salloc is successful
    if node_range_length == 1:
        # only one HPC node is being used
        #
        # This is a wrong assignment: cosim_slurm_nodes['CO_SIM_SLURM_NODE_000'] = slurm_node_list_prefix_and_range[0]
        #
        # cosim_slurm_nodes['CO_SIM_SLURM_NODE_000'] = \
        #     {constants.CO_SIM_VARIABLE_DESCRIPTION: 'SLURM compute node hostname',
        #      constants.CO_SIM_VARIABLE_VALUE: slurm_node_list_prefix_and_range[0] }
        cosim_slurm_nodes['CO_SIM_SLURM_NODE_000'] = slurm_node_list_prefix_and_range[0]
    else:
        # two or more HPC nodes have been allocated
        hpc_nodes_name_prefix = slurm_node_list_prefix_and_range[0]
        hpc_nodes_name_range = slurm_node_list_prefix_and_range[1]
        nodes_name_suffix_list = re.split(r'-', hpc_nodes_name_range)
        first_node_name_suffix = nodes_name_suffix_list[0]
        last_node_name_suffix = nodes_name_suffix_list[1]

        n_correlative = 0
        for curr_n_node_name_suffix in range(int(first_node_name_suffix), int(last_node_name_suffix) + 1):
            co_sim_slurm_node_variable_name = f'CO_SIM_SLURM_NODE_{n_correlative:0>3d}'
            cosim_slurm_nodes[co_sim_slurm_node_variable_name] =\
                f'{hpc_nodes_name_prefix}{curr_n_node_name_suffix:0>3d}'

            n_correlative += 1

    return cosim_slurm_nodes
