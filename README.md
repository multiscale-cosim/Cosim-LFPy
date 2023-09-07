<div align="center" id="top"> 
  <img src="../../../misc/logo.jpg" alt="Modular Science: Multi-scale Co-simulation" />

  &#xa0;
</div>

<h1 align="center">Modular Science: Multi-scale Co-simulation</h1>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />

  <img alt="Github language count" src="https://img.shields.io/github/languages/count/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />

  <img alt="License" src="https://img.shields.io/github/license/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />

  <img alt="Github issues" src="https://img.shields.io/github/issues/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />

  <img alt="Github forks" src="https://img.shields.io/github/forks/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />

  <img alt="Github stars" src="https://img.shields.io/github/stars/multiscale-cosim/TVB-NEST-usecase1?color=56BEB8" />
</p>

<h4 align="center"> 
	Multi-scale Co-simulation - NEST-LFPy
</h4> 

<hr>

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Getting Started</a> &#xa0; | &#xa0;
  <a href="#memo-license">License</a> &#xa0; | &#xa0;
  <a href="https://github.com/multiscale-cosim" target="_blank">Author</a> &#xa0; | &#xa0;
  <a href="https://github.com/multiscale-cosim" target="_blank">Acknowledgement</a>
</p>

<br>

## About ##

This usecase demonstrates how to use the Co-simulation framework to calculate Local Field Potentials (LFPs) in real time, based on spike events streamed from the NEST simulator. 

We used the cortical microcircuit model by Potjans and Diesmann, available from
https://nest-simulator.readthedocs.io/en/stable/auto_examples/Potjans_2014/

The LFP signals are calculated from the spike events by applying the so-called kernel approach, as outlined in Hagen et al. (2022):
https://doi.org/10.1371/journal.pcbi.1010353

This usecase can be used as a starting point for applying the Co-simulation framework to other network models in NEST or TVB, and also for simulating other brain signals like EEG, MEG, or ECoG signals.   

## Technologies ##

The following tools were used in this project:

- [Python](https://www.python.org/)
- [CMake](https://cmake.org/)
- [C++](https://isocpp.org/)
- [Makefile](https://www.gnu.org/software/make/manual/make.html)

## Getting Started ##

The Modular Science Co-simulaiton framework and usecase can be installed and launched on:
- **Local systems:** e.g. a virtual machine (VM) on a laptop. We support the useage of Virtualbox and Vagrant.
<!-- - **HPC systems:** currently supported on the [JUWELS](https://apps.fz-juelich.de/jsc/hps/juwels/index.html) and [JUSUF](https://apps.fz-juelich.de/jsc/hps/jusuf/index.html) clusters at the Jülich Supercomputing Centre.

The intended platform to deploy the MSC framework with this co-simulation usecase are HPC systems.
They allow independant scaling of the components and efficient simulations. Deploying it on a laptop aids testing and development. -->

### Installation ###

Please check [HERE](https://github.com/multiscale-cosim/Cosim-LFPy/tree/main/INSTALL.md) for installation details.


### How to run ###
 
 The framework and usecase can be installed and launched on:
- **Local systems:**
  - go to [run_usecase/local](https://github.com/multiscale-cosim/Cosim-LFPy/tree/main/run_usecase/local) directory and run [this](https://github.com/multiscale-cosim/Cosim-LFPy/blob/main/run_usecase/local/run_on_local.sh) script from there e.g.

  ```
  $ sh ./run_on_local.sh
  ```
  - On the VM (see Installation guide above) you will find two scripts in `/home/vagrant/multiscale-cosim`:
    ```
    vagrant@ubuntu-focal:~multiscale-cosim$ source Cosim-LFPy.source
    vagrant@ubuntu-focal:~multiscale-cosim$ ./run_on_loacl.sh
    ```

<!-- - **HPC systems:** To execute the usecase on HPC systems, go to [run_usecase/hpc](https://github.com/multiscale-cosim/TVB-NEST-usecase1/tree/hpc/run_usecase/hpc) directory. The usecase can be deployed and executed within an interactive session or could also be submitted as a SLURM job. 

  - Interactive session: first allocate the required resources by specifying the cluster partition and account e.g:
    ```
    $ salloc --partition=<partition> --nodes=2 --account=<account>
    ```
    Then, run [this](https://github.com/multiscale-cosim/TVB-NEST-usecase1/blob/hpc/run_usecase/hpc/cosim_launch_hpc_sbatch.sh) script from there e.g.:

    ```
    $ sh ./cosim_launch_hpc_sbatch.sh
    ```

  - SLURM job: To submit the usecase as a slurm job, run [this](https://github.com/multiscale-cosim/TVB-NEST-usecase1/blob/hpc/run_usecase/hpc/run_usecase_sbatch.sh) script e.g.:

    ```
    $ sh ./run_usecase_sbatch.sh
    ```

    **NOTE** It will create a directory named as _slurm_logs_ at the [same location](https://github.com/multiscale-cosim/TVB-NEST-usecase1/tree/hpc/run_usecase/hpc) where the execution script is run, to capture the outputs and errors from the SLURM. -->

  **Simulation Results:** The simulation results, logs, and the resource usage stats can be found in directory ***Cosimulation_outputs*** created by Modular Science during the execution at the [same location](https://github.com/multiscale-cosim/Cosim-LFPy/tree/main/run_usecase/local) where the execution script is run.
-- --

## License ##

This project is under license from Apache License, Version 2.0. For more details, see the [LICENSE](LICENSE) file.


Made by <a href="https://github.com/multiscale-cosim" target="_blank">Multiscale Co-simulation team</a>.

## Acknowledgement ##

This project has received funding from the European Union’s Horizon 2020 research and innovation
programme under grant agreement No 785907 (HBP SGA2), from the European Union’s Horizon
2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No.
945539 (HBP SGA3) and from the European Union’s Horizon 2020 Framework Programme for
Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project
SGA3)


&#xa0;

<a href="#top">Back to top</a>
