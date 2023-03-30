# Modular Science - Multisclae Co-simulation with LFPy usecase
### WIP
---
#### Get repositories and install everything
  - For VM it is the same process as for TVB-NEST usecase: https://github.com/multiscale-cosim/TVB-NEST-usecase1/blob/main/INSTALL.md
      - updated `bootstrap.sh` and `Vagrantfile` in https://github.com/multiscale-cosim/Cosim-LFPy/tree/main/installation/local

Short description:
  - Have virtualbox and vagrant installed
  - get the bootstrap.sh and Vagrantfile from the installation folder
    - create a VM directory on the host-system with the bootrap.sh, Vagrantfile and a shared directory
  - run `vagrant up` to start the VM with LFPy and NEST installed
  - run `vagrant ssh` to log in to the VM
  - test the installation:
    - `cd multiscale-cosim`
    - `source Cosim-LFPy.source`
    - `python3 Cosim-LFPy/installation/tests/nest_test.py`
    - TODO add installation test script for LFPy

---
### Next steps / TODOs:
#### Action Adapters
  - Interscalehub (copied from TVB-Nest usecase)
  - Nest (copied from TVB-Nest usecase): new adapter / edit current one to fit this usecase
     - Create a Nest adapter to run Potjans_2014->run_microcircuit.py. For inspiration, look (https://github.com/mfahdaz/nest-simulator/blob/ecc373fe83a82d133f562a4ed6cf180d07c39578/pynest/examples/Potjans_2014/run_microcircuit.py)
     - NOTE: its one-way simulation - no MPI connection for spike generators
#### InterscaleHUB
  - New communicator in InterscaleHub to
     - receive the data from NEST
     - do LFPy after receiving the data from NEST see e.g. line 228 in (https://github.com/multiscale-cosim/EBRAINS_InterscaleHUB/blob/bd974bd6a3ea809dbdee6cf77c9082c81192ca80/Interscale_hub/communicator_nest_to_tvb.py#L228).
     - save the results to file
  -Nothing to send back to NEST as a first step maybe? For the next step maybe we need to create some filter node (from NESTML)/multimeter instead of spike generator e.g.  line 196 (https://github.com/multiscale-cosim/TVB-NEST-usecase1/blob/247c1f59ca576fd842aee02dd408f7d13aae6d63/action_adapters_alphabrunel/nest_simulator/nest_adapter.py#L196)

#### LFPy kernels
  - Create LFPy functionality to add in InterscaleHub
#### Configurations files
  - Actions, Parameters, Plans
  - Create new ones or extend current ones?
  - for local and/or hpc systems ?
