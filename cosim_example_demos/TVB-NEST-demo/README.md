## Changelog for cosim template demo
* transformation with InterscaleHUB
* temporary result folder creation and deletion upon run in run.py
* removed launcher subfolder
* removed transformation subfolder
* neccessary changes of relative paths in run.py and parameter.py 
* moved old demo files and cleanup


# TVB-NEST-demo
temporary repo for a TVB-NEST demonstrator

## Topic
This repository is a demonstration for the co-simulation between TVB, NEST and Elephant.
It should be flexible and scalable to adapt to any networks simulation and run in supercomputers. <br />
It's a demonstration of a work underdevelopment: https://github.com/multiscale-cosim/TVB-NEST

## Table of Contents
1. [Installation](#installation)
2. [Example](#example)
3. [Files](#files)

## Installation and update :<a name="installalation"></a>
For the installation, you should look at the script in the folder install. The docker file is a good example 
of finding of the missing dependency.<br />
The ubuntu_installation.sh is an incomplete script to install all the tools.

## Example<a name="example"></a>
This demonstration is based on the example of NEST and TVB. It can help you to develop your own co-simulation 
but it is not optimized and it can miss some functionalities.<br />
For running the example, run the script nest_elephant_tvb/launcher/run.py

## Files<a name="files"></a>
* install: Documentation of the project
* nest_elephant_tvb: file which contains all the kernel of the simulation
    * launcher:
        * run.py: main script of the simulation on one simulator or the co-simulation. 
    * nest: folder contains all the file to configure and launch Nest
        * Balanced_network_reduce_co-sim.sh: run the NEST example 
        * utils_function.py: function used for the Nest simulation 
    * transformation: folder contains the transformer between TVB and Nest using Elephant
        * communication: functions for internal ad external communication
            * internal.py: abstract classes for the internal communication/ the communication between component of the transformer 
            * internal_mpi.py: implementation with MPI communication
            * internal_thread.py: implementation with thread communication
            * mpi_io_external.py: management the mpi communication with the simulator
        * simulator: Communication with simulator
            * Nest_IO: communication with NEST simulator
            * TVB_IO: communication with TVB simulator
        * transformation_function: function for the transformation
            * abstract_transformation_function.py: Management communication with the other components of the transformer
            * transformation_function.py: implementation of the transformation function
        * nest_to_tvb: launcher of the component of the transformer between NEST to TVB
        * tvb_to_nest: launcher of the component of the transformer between TVB to NEST
    * TVB
        * TVB_simple_example_co_sim.sh: run the TVB example
        * wrapper_TVB_mpi.py: wrapper of TVB for management of the MPI communication.
    *utils.py: functions shared by all the modules

