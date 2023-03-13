#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "

import datetime
import os
import json
import subprocess
import copy
from nest_elephant_tvb.utils import create_folder, create_logger


def run(parameters):
    '''
    run the simulation
    :param parameters: parameters of the simulation
    :return:
    '''
    path = parameters['path']
    # start to create the repertory for the simulation
    create_folder(path)
    create_folder(path + "/log")
    create_folder(path + '/nest')
    create_folder(path + '/tvb')
    create_folder(path + '/transformation')
    create_folder(path + '/transformation/spike_detector/')
    create_folder(path + '/transformation/send_to_tvb/')
    create_folder(path + '/transformation/spike_generator/')
    create_folder(path + '/transformation/receive_from_tvb/')
    create_folder(path + '/figures')
    save_parameter(parameters)

    logger = create_logger(path, 'launcher', parameters['level_log'])

    logger.info('time: ' + str(datetime.datetime.now()) + ' BEGIN SIMULATION \n')

    # chose between running on cluster or local pc
    mpirun = ['mpirun']  # example : ['mpirun'] , ['srun','-N','1']

    processes = []  # process generate for the co-simulation
    processes.append(run_nest(mpirun, parameters['path'] + '/parameter.json', logger))

    # create transformer between Nest to TVB :
    processes.append(run_nest_to_tvb(mpirun, parameters['path'], logger))

    # create transformer between TVB to Nest:
    processes.append(run_tvb_to_nest(mpirun, parameters['path'], logger))

    # Run TVB in co-simulation
    processes.append(run_tvb(mpirun, parameters['path'] + '/parameter.json', logger))

    # FAT END POINT : add monitoring of the different process
    for process in processes:
        process.wait()
    logger.info('time: ' + str(datetime.datetime.now()) + ' END SIMULATION \n')


def run_nest(mpirun, path_parameter, logger):
    """
    launch NEST
    :param mpirun: multiprocessor launcher
    :param path_parameter: path of the parameter file
    :param logger: logger of the launcher
    :return:
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../nest/Balanced_network_reduce_co-sim.py"
    argv = copy.copy(mpirun)
    argv += ['-n', '2', 'python3', dir_path]
    argv += [path_parameter]
    logger.info("NEST start :" + str(argv))
    return subprocess.Popen(argv,
                            # need to check if it's needed or not (doesn't work for me)
                            stdin=None, stdout=None, stderr=None, close_fds=True,  # close the link with parent process
                            )


def run_tvb(mpirun, path_parameter, logger):
    """
    launch TVB
    :param mpirun: multiprocessor launcher
    :param path_parameter: path of the parameter file
    :param logger: logger of the launcher
    :return:
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../tvb/TVB_simple_example_co_sim.py"
    argv = copy.copy(mpirun)
    argv += ['-n', '1', 'python3', dir_path]
    argv += [path_parameter]
    logger.info("TVB start :" + str(argv))
    print("TVB start :", argv)
    return subprocess.Popen(argv,
                            # need to check if it's needed or not (doesn't work for me)
                            stdin=None, stdout=None, stderr=None, close_fds=True,  # close the link with parent process
                            )


def run_nest_to_tvb(mpirun, path, logger):
    """
    launch Transformer NEST to TVB
    :param mpirun: multiprocessor launcher
    :param path: path of the simulation folder
    :param logger: logger of the launcher
    :return:
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../transformation/nest_to_tvb.py"
    argv = copy.copy(mpirun)
    argv += ['-n', '3', 'python3', dir_path]
    argv += [path]
    logger.info("Transformer NEST to TVB start : " + str(argv))
    return subprocess.Popen(argv,
                            # need to check if it's needed or not (doesn't work for me)
                            stdin=None, stdout=None, stderr=None, close_fds=True,  # close the link with parent process
                            )


def run_tvb_to_nest(mpirun, path, logger):
    """
    launch Transformer TVB to NEST
    :param mpirun: multiprocessor launcher
    :param path: path of the simulation folder
    :param logger: logger of the launcher
    :return:
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../transformation/tvb_to_nest.py"
    argv = copy.copy(mpirun)
    argv += ['-n', '3', 'python3', dir_path]
    argv += [path]
    logger.info("Translator TVB to NEST start : " + str(argv))
    return subprocess.Popen(argv,
                            # need to check if it's needed or not (doesn't work for me)
                            stdin=None, stdout=None, stderr=None, close_fds=True,  # close the link with parent process
                            )


def save_parameter(parameters):
    """
    save the parameters of the simulations in json file
    :param parameters: dictionary of parameters
    :return: nothing
    """
    # save the value of all parameters
    f = open(parameters['path'] + '/parameter.json', "wt")
    json.dump(parameters, f)
    f.close()


if __name__ == "__main__":
    parameter_default = {"co_simulation": False,
                         "path": "",
                         "simulation_time": 100.0,
                         "level_log": 1,
                         "resolution": 0.1,
                         "nb_neurons": [100]
                         }

    # NEST only
    path_file = os.path.dirname(__file__)
    parameter_nest_only = copy.copy(parameter_default)
    parameter_nest_only['path'] = path_file + "/../../result_sim/nest_only/"
    create_folder(parameter_nest_only['path'])
    create_folder(parameter_nest_only['path'] + "/log/")
    create_folder(parameter_nest_only['path'] + "/figures/")
    create_folder(parameter_nest_only['path'] + "/nest/")
    save_parameter(parameter_nest_only)
    logger_nest_only = create_logger(parameter_nest_only['path'], 'launcher', 1)
    process = run_nest(['mpirun'], parameter_nest_only['path'] + '/parameter.json', logger_nest_only)
    process.wait()

    # TVB only
    path_file = os.path.dirname(__file__)
    parameter_tvb_only = copy.copy(parameter_default)
    parameter_tvb_only['path'] = path_file + "/../../result_sim/tvb_only/"
    create_folder(parameter_tvb_only['path'])
    create_folder(parameter_tvb_only['path'] + "/log/")
    create_folder(parameter_tvb_only['path'] + "/figures/")
    save_parameter(parameter_tvb_only)
    logger_tvb_only = create_logger(parameter_tvb_only['path'], 'launcher', 1)
    process = run_tvb(['mpirun'], parameter_tvb_only['path'] + '/parameter.json', logger_tvb_only)
    process.wait()

    # Co-simulation
    path_file = os.path.dirname(__file__)
    parameter_co_simulation = copy.copy(parameter_default)
    parameter_co_simulation['path'] = path_file + "/../../result_sim/co-simulation/"
    parameter_co_simulation.update({
        "co_simulation": True,
        # parameter for the synchronization between simulators
        "time_synchronization": 1.2,
        "id_nest_region": [0],
        # parameter for the transformation of data between scale
        "nb_brain_synapses": 1,
        'id_first_neurons': [1],
        "save_spikes": True,
        "save_rate": True,
    })
    run(parameter_co_simulation)
