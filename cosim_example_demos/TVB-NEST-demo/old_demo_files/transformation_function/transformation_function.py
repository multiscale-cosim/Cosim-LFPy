#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np
from neo.core import SpikeTrain, AnalogSignal
from quantities import ms, Hz
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.statistics import instantaneous_rate
from elephant.kernels import RectangularKernel
from nest_elephant_tvb.transformation.transformation_function.abstract_transformation_function import \
    AbstractTransformationRateSpike, AbstractTransformationSpikeRate


class TransformationSpikeRate(AbstractTransformationSpikeRate):
    """
    Implementation of the transformation function spike trains to rate
    """

    def __init__(self, id_transformer, param, *arg, **karg):
        """
        see super class
        get the parameter specific for the transformation function
        """
        super().__init__(id_transformer, param, *arg, **karg)
        self.nb_neurons = param['nb_neurons'][id_transformer]
        self.first_id = param['id_first_neurons'][id_transformer]

    def spike_to_rate(self, count, size_buffer, buffer_of_spikes):
        """
        function for the transformation of the spike trains to rate
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param size_buffer: size of the data in the buffer
        :param buffer_of_spikes: buffer contains spikes
        :return: rate for the interval
        """
        spikes_neurons = self._reshape_buffer_from_nest(count, size_buffer, buffer_of_spikes)
        rates = instantaneous_rate(spikes_neurons,
                                   t_start=np.around(count * self.time_synch, decimals=2) * ms,
                                   t_stop=np.around((count + 1) * self.time_synch, decimals=2) * ms,
                                   sampling_period=(self.dt - 0.000001) * ms, kernel=RectangularKernel(1.0 * ms))
        rate = np.mean(rates, axis=1) / 10  # the division by 10 ia an adaptation for the model of TVB
        times = np.array([count * self.time_synch, (count + 1) * self.time_synch], dtype='d')
        return times, rate

    def _reshape_buffer_from_nest(self, count, size_buffer, buffer):
        """
        get the spike time from the buffer and order them by neurons
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param size_buffer: size of the data in the buffer
        :param buffer: buffer contains id of devices, id of neurons and spike times
        :return:
        """
        spikes_neurons = [[] for i in range(self.nb_neurons)]
        # get all the time of the spike and add them in a histogram
        for index_data in range(int(np.rint(size_buffer / 3))):
            id_neurons = int(buffer[index_data * 3 + 1])
            time_step = buffer[index_data * 3 + 2]
            spikes_neurons[id_neurons - self.first_id].append(time_step)
        for i in range(self.nb_neurons):
            if len(spikes_neurons[i]) != 0:
                spikes_neurons[i] = SpikeTrain(np.concatenate(spikes_neurons[i]) * ms,
                                               t_start=np.around(count * self.time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.time_synch, decimals=2) + 0.0001)
            else:
                spikes_neurons[i] = SpikeTrain(spikes_neurons[i] * ms,
                                               t_start=np.around(count * self.time_synch, decimals=2),
                                               t_stop=np.around((count + 1) * self.time_synch, decimals=2))
        return spikes_neurons


class TransformationRateSpike(AbstractTransformationRateSpike):
    """
    Implementation of the transformation function rate to spike trains
    """
    def __init__(self, id_translator, param, nb_spike_generator, *arg, **karg):
        super().__init__(id_translator, param, nb_spike_generator, *arg, **karg)
        self.nb_synapse = param["nb_brain_synapses"]

    def rate_to_spike(self, count, time_step, rate):
        """
        function for the transformation of the data
        :param count: counter of the number of time of the transformation (identify the timing of the simulation)
        :param time_step: time of transformation
        :param rate: rate to transform
        :return: array of spike trains
        """
        # Single Interaction Process Model
        # Compute the rate to spike trains
        rate *= self.nb_synapse  # rate of poisson generator ( due property of poisson process)
        rate += 1e-12
        rate = np.abs(rate)  # avoid rate equals to zeros
        signal = AnalogSignal(rate * Hz, t_start=(time_step[0] + 0.1) * ms,
                              sampling_period=(time_step[1] - time_step[0]) / rate.shape[-1] * ms)
        spike_generate = []
        for i in range(self.nb_spike_generator):
            # generate individual spike trains
            spike_generate.append(np.around(np.sort(inhomogeneous_poisson_process(signal, as_array=True)), decimals=1))
        return spike_generate
