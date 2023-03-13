#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "

import sys
import numpy as np
from mpi4py import MPI
import os
import time


def run_mpi(simulator, path, logger):
    """
    return the result of the simulation between the wanted time
    :param simulator: tvb simulator
    :param path: the folder of the simulation
    :param logger: logger of the run
    """
    end = simulator.simulation_length
    dt = simulator.integrator.dt
    time_synch = simulator.synchronization_time
    time_synch_n = int(np.around(time_synch / dt))
    nb_monitor = len(simulator.monitors)
    id_proxy = simulator.proxy_inds

    # path = "/home/vagrant/multiscale-cosim-repos/TVB-NEST-demo/result_sim/co-simulation"
    path_send = path + "/transformation/send_to_tvb/"
    path_receive = path + "/transformation/receive_from_tvb/"
    # path_send = "/home/vagrant/multiscale-cosim-repos/TVB-NEST-demo/result_sim/co-simulation/transformation/send_to_tvb/"
    # path_receive = "/home/vagrant/multiscale-cosim-repos/TVB-NEST-demo/result_sim/co-simulation//transformation/receive_from_tvb/"

    # initialise the variable for the saving the result
    save_result = []
    for i in range(nb_monitor):  # the input output monitor
        save_result.append([])

    # init MPI :
    comm_receive = []
    for i in id_proxy:
        comm_receive.append(init_mpi(path_send + str(i) + ".txt", logger))
    comm_send = []
    for i in id_proxy:
        comm_send.append(init_mpi(path_receive + str(i) + ".txt", logger))

    logger.info("send initialisation of TVB : prepare data")
    initialisation_data = []
    for i in np.arange(0, time_synch_n, 1, dtype=np.int):
        initialisation_data.append(simulator._loop_compute_node_coupling(i)[:, id_proxy, :])
    initialisation_data = np.concatenate(initialisation_data)
    time_init = [0, time_synch]
    logger.info("send initialisation of TVB : send data")
    for index, comm in enumerate(comm_send):
        send_mpi(comm, time_init, initialisation_data[:, index] * 1e3, logger)

    # the loop of the simulation
    count = 0
    while count * time_synch < end:  # FAT END POINT
        logger.info(" TVB receive data start")
        # receive MPI data
        data_value = []
        for comm in comm_receive:
            receive = receive_mpi(comm, logger)
            time_data = receive[0]
            data_value.append(receive[1])
        logger.info(" TVB receive data values")
        data = np.empty((2,), dtype=object)
        nb_step = np.rint((time_data[1] - time_data[0]) / dt)
        nb_step_0 = np.rint(time_data[0] / dt) + 1  # start at the first time step not at 0.0
        time_data = np.arange(nb_step_0, nb_step_0 + nb_step, 1) * dt
        data_value = np.swapaxes(np.array(data_value), 0, 1)[:, :]
        data_value = np.expand_dims(data_value, axis=(1, 3))
        if data_value.shape[0] != time_data.shape[0]:
            print(nb_step)
            print(dt, time_data, receive)
            raise (Exception('Bad shape of data ' + str(data_value.shape[0]) + " " + str(time_data.shape[0])))
        data[:] = [time_data, data_value]

        logger.info(" TVB start simulation " + str(count * time_synch))
        for result in simulator(simulation_length=time_synch, cosim_updates=data):
            for i in range(nb_monitor):
                if result[i] is not None:
                    save_result[i].append(result[i])

        logger.info(" TVB end simulation")

        # prepare to send data with MPI
        nest_data = simulator.loop_cosim_monitor_output(n_steps=time_synch_n)[0]
        times = [nest_data[0][0], nest_data[0][-1]]
        rate = np.concatenate(nest_data[1][:, 0, [id_proxy], 0])
        for index, comm in enumerate(comm_send):
            send_mpi(comm, times, rate[:, index] * 1e3, logger)

        # increment of the loop
        count += 1
    # save the last part
    logger.info(" TVB finish")
    for index, comm in enumerate(comm_send):
        logger.info('end comm send')
        end_mpi(comm, path + "/transformation/receive_from_tvb/" + str(id_proxy[index]) + ".txt", True, logger)
    for index, comm in enumerate(comm_receive):
        logger.info('end comm receive')
        end_mpi(comm, path + "/transformation/send_to_tvb/" + str(id_proxy[index]) + ".txt", False, logger)
    MPI.Finalize()  # ending with MPI
    logger.info(" TVB exit")
    return reshape_result(save_result)


## MPI function for receive and send data
def init_mpi(path, logger):
    """
    initialise MPI connection
    :param path: path of the file for the port
    :param logger: logger of the modules
    :return:
    """
    while not os.path.exists(path + '.unlock'):  # FAT END POINT
        logger.info(path + '.unlock')
        logger.info("spike detector ids not found yet, retry in 1 second")
        time.sleep(1)
    os.remove(path + '.unlock')
    fport = open(path, "r")
    port = fport.readline()
    fport.close()
    logger.info("wait connection " + port)
    comm = MPI.COMM_WORLD.Connect(port)
    logger.info('connect to ' + port)
    return comm


def send_mpi(comm, times, data, logger):
    """
    send mpi data
    :param comm: MPI communicator
    :param times: times of values
    :param data: rates inputs
    :param logger: logger of the modules
    :return:nothing
    """
    logger.info("start send")
    status_ = MPI.Status()
    # wait until the transformer accept the connections
    accept = False
    while not accept:
        req = comm.irecv(source=0, tag=0)
        accept = req.wait(status_)
        logger.info("send accept")
    source = status_.Get_source()  # the id of the excepted source
    logger.info("get source")
    data = np.ascontiguousarray(data, dtype='d')  # format the rate for sending
    shape = np.array(data.shape[0], dtype='i')  # size of data
    times = np.array(times, dtype='d')  # time of starting and ending step
    comm.Send([times, MPI.DOUBLE], dest=source, tag=0)
    comm.Send([shape, MPI.INT], dest=source, tag=0)
    comm.Send([data, MPI.DOUBLE], dest=source, tag=0)
    logger.info("end send")


def receive_mpi(comm, logger):
    """
        receive proxy values the
    :param comm: MPI communicator
    :param logger: logger of the modules
    :return: rate of all proxy
    """
    logger.info("start receive")
    status_ = MPI.Status()
    # send to the transformer : I want the next part
    req = comm.isend(True, dest=0, tag=0)
    req.wait()
    time_step = np.empty(2, dtype='d')
    comm.Recv([time_step, 2, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
    # get the size of the rate
    size = np.empty(1, dtype='i')
    comm.Recv([size, MPI.INT], source=0, tag=0)
    # get the rate
    rates = np.empty(size, dtype='d')
    comm.Recv([rates, size, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
    logger.info("end receive " + str(time_step))
    # print the summary of the data
    if status_.Get_tag() == 0:
        return time_step, rates
    else:
        return None


def end_mpi(comm, path, sending, logger):
    """
    ending the communication
    :param comm: MPI communicator
    :param path: for the close the port
    :param sending: if the transformer is for sending or receiving data
    :param logger: logger of the module
    :return: nothing
    """
    # path = "/home/vagrant/multiscale-cosim-repos/TVB-NEST-demo/result_sim/co-simulation"
    # read the port before the deleted file
    fport = open(path, "r")
    port = fport.readline()
    fport.close()
    # different ending of the transformer
    if sending:
        logger.info("TVB close connection send " + port)
        sys.stdout.flush()
        status_ = MPI.Status()
        # wait until the transformer accept the connections
        logger.info("TVB send check")
        accept = False
        while not accept:
            req = comm.irecv(source=0, tag=0)
            accept = req.wait(status_)
        logger.info("TVB send end simulation")
        source = status_.Get_source()  # the id of the excepted source
        times = np.array([0., 0.], dtype='d')  # time of starting and ending step
        comm.Send([times, MPI.DOUBLE], dest=source, tag=1)
    else:
        logger.info("TVB close connection receive " + port)
        # send to the transformer : I want the next part
        req = comm.isend(True, dest=0, tag=1)
        req.wait()
    # closing the connection at this end
    logger.info("TVB disconnect communication")
    comm.Disconnect()
    logger.info("TVB close " + port)
    MPI.Close_port(port)
    logger.info("TVB close connection " + port)
    return

## Function for reshape the result of TVB
def reshape_result(result):
    """
    reshape the output of TVB for the
    :param result: output of TVB
    :return: Jülich Supercomputing Centre
    """
    times = []
    values = []
    for (time, value) in result[0]:
        if time > 0.0:
            times.append(time)
            values.append(value)
    return ([np.array(times), np.expand_dims(np.concatenate(values), 1)],)
