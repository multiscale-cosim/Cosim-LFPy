#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "

import numpy as np
from nest_elephant_tvb.transformation.communication.mpi_io_external import MPICommunicationExtern


class AbstractTransformationSpikeRate(MPICommunicationExtern):
    """
    Class for the transformation spikes to rate
    """

    def __init__(self, id_transformer, param, *arg, **karg):
        """
        transformation object from spikes to rate
        :param id_transformer : id of the transformer
        :param param: parameter of the translation function
        :param arg: parameters
        :param karg: other parameters
        """
        super().__init__(*arg, **karg)
        self.id = id_transformer
        self.time_synch = param['time_synchronization']  # time of synchronization between 2 run
        self.dt = param['resolution']  # the resolution of the integrator
        self.path = param['path'] + "/transformation/"
        # variable for saving values:
        self.save_spikes = bool(param['save_spikes'])
        if self.save_spikes:
            self.save_spikes_buf = None
        self.save_rate = bool(param['save_rate'])
        if self.save_rate:
            self.save_rate_buf = None

    def simulation_time(self):
        """
        Translation function of the spike to rate :
            1) get the spike
            2) transform spike to rate
            3) send rate
        The step 1 and 3 need to be dissociate for synchronization requirement.
        This dissociation allow the transformation module to buffer one more step from the sender or the receiver.
        This function is very important for the speed of the simulation
        """
        # initialisation of the communication
        self.logger.info('TRS : begin sim')
        rates = np.zeros((int(self.time_synch / (self.dt - 0.000001)), 1))  # initialisation of the communication with 0
        self.logger.info('TRS : init rates')
        self.communication_internal.send_time_rate(np.array([0., self.time_synch]), rates)
        self.logger.info('TRS : send init')
        count = 0  # counter of the number of run. It can be useful for the transformation function
        while True:
            # Step 1: INTERNAL : get spike
            self.logger.info('TSR : receive data Nest')
            self.communication_internal.get_spikes_ready()
            if self.communication_internal.shape_buffer[0] == -1:
                self.logger.info('TSR : break')
                break

            # optional : save spikes
            if self.save_spikes:
                if self.save_spikes_buf is None:
                    self.save_spikes_buf = \
                        np.copy(self.communication_internal.databuffer[:self.communication_internal.shape_buffer[0]])
                else:
                    self.save_spikes_buf = \
                        np.concatenate((self.save_spikes_buf,
                                        np.copy(self.communication_internal.databuffer[
                                                :self.communication_internal.shape_buffer[0]])))

            # Step 2.1: take all data from buffer and compute rate
            self.logger.info('TSR : add spikes ' + str(self.communication_internal.shape_buffer[0]))
            times, rate = self.spike_to_rate(count,
                                             self.communication_internal.shape_buffer[0],
                                             self.communication_internal.databuffer)

            # Step 2.2 : INTERNAL: end get spike (important to be here for optimization/synchronization propose)
            self.communication_internal.get_spikes_release()

            # optional : save rate
            if self.save_rate:
                if self.save_rate_buf is None:
                    self.save_rate_buf = rate
                else:
                    self.save_rate_buf = np.concatenate((self.save_rate_buf, rate))

            # Step 3: INTERNAL: send rate and time
            self.logger.info('TSR : send data')
            self.communication_internal.send_time_rate(times, rate)
            if self.communication_internal.send_time_rate_exit:
                self.logger.info('TSR : break 2')
                break

            # Step 4 : end loop
            count += 1
        # INTERNAL: Close all the internal communication
        self.logger.info('TSR : end methods')
        self.communication_internal.get_spikes_end()
        self.communication_internal.send_time_rate_end()
        self.logger.info('TSR : end')

    def finalise(self):
        """
        see super class
        """
        super().finalise()
        # Save the ending part of the simulation
        if self.save_spikes:
            np.save(self.path + '/spikes_' + str(self.id) + '.npy', self.save_spikes_buf)
        if self.save_rate:
            np.save(self.path + '/rates_' + str(self.id) + '.npy', self.save_rate_buf)

    def spike_to_rate(self, count, size_buffer, buffer_of_spikes):
        """
        function for the transformation of the data
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param size_buffer: size of the data in the buffer
        :param buffer_of_spikes: buffer contains spikes
        :return: rate for the interval
        """
        raise Exception('not implemented')


class AbstractTransformationRateSpike(MPICommunicationExtern):
    """
    Class for the transformation between rate to spike
    """

    def __init__(self, id_transformer, param, nb_spike_generator, *arg, **karg):
        """
        translation from rate to spike trains
        :param id_transformer: id of the transformer
        :param param: parameter for the transformation function
        :param nb_spike_generator: number of spike generators
        :param arg: parameters
        :param karg: other parameters
        """
        super().__init__(*arg, **karg)
        self.id = id_transformer
        self.nb_spike_generator = nb_spike_generator  # number of spike generator
        self.path = param['path'] + "/transformation/"
        # variable for saving values:
        self.save_spike = bool(param['save_spikes'])
        if self.save_spike:
            self.save_spike_buf = None
        self.save_rate = bool(param['save_rate'])
        if self.save_rate:
            self.save_rate_buf = None
        self.logger.info('TRS : end init transformation')

    def simulation_time(self):
        """
        Translation function of the rate to spike :
            1) get the rate
            2) transform rate to spike
            3) send spike trains
        The step 1 and 3 need to be dissociate for synchronization requirement.
        This dissociation allow the transformation module to buffer one more step from the sender or the receiver.
        This function is very important for the speed of the simulation.
        """
        count = 0  # counter of the number of run. It use to send the good time to TVB
        while True:
            # Step 1.1: INTERNAL : get rate
            self.logger.info('TRS : get rate')
            times, rate = self.communication_internal.get_time_rate()
            if self.communication_internal.get_time_rate_exit:
                self.logger.info('TRS : break end sender')
                break

            # Step 1.2: INTERNAL : end getting rate (important to be here for optimization/synchronization propose)
            self.communication_internal.get_time_rate_release()

            # optional :  save the rate
            if self.save_rate:
                if self.save_rate_buf is None:
                    self.save_rate_buf = rate
                else:
                    self.save_rate_buf = np.concatenate((self.save_rate_buf, rate))

            # Step 2: generate spike trains
            # improvement : we can generate other type of data but Nest communication need to be adapted for it
            self.logger.info('TRS : generate spike')
            spike_trains = self.rate_to_spike(count, times, rate)

            # optional : save spikes
            if self.save_spike:
                if self.save_spike_buf is None:
                    self.save_spike_buf = [spikes for spikes in spike_trains]
                else:
                    tmp = []
                    for index, spikes in enumerate(spike_trains):
                        tmp.append(np.concatenate((self.save_spike_buf[index], spikes)))
                    self.save_spike_buf = tmp

            # Step 3: send spike trains to Nest
            self.logger.info('TRS : send spike train')
            self.communication_internal.send_spikes_trains(spike_trains)
            if self.communication_internal.send_spike_exit:
                self.logger.info('TRS : break')
                break

            # Step 4 : end loop
            count += 1
        # INTERNAL: Close all the internal communication
        self.logger.info('TRS : end method by TVB : ' + str(self.communication_internal.get_time_rate_exit))
        self.communication_internal.get_time_rate_end()
        self.communication_internal.send_spikes_end()
        self.logger.info('TRS : end')

    def finalise(self):
        """
        see super class
        :return:
        """
        super().finalise()
        # Save the ending part of the simulation
        if self.save_rate:
            np.save(self.path + '/rate_' + str(self.id) + '.npy', self.save_rate_buf)
        if self.save_spike:
            np.save(self.path + '/spike_' + str(self.id) + '.npy', self.save_spike_buf)

    def rate_to_spike(self, count, time_step, rate):
        """
        function for the transformation of the data
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param time_step: time of transformation
        :param rate: rate to transform
        :return: array of spike trains
        """
        raise Exception('not implemented')
