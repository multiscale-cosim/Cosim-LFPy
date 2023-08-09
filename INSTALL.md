# Installation guide for the MSC framework and NEST-LFPy usecase

This installation instructions are intended to minimize changes to the local environment. On local systems the use of a virtual machine ensures no conflicting dependencies. On the supported HPC systems, the use of pre-installed modules is leveraged.

* Platforms
  <!-- 1. HPC systems -(#Cosim-NEST-LFPy usecase-installation-on-HPC-systems) -->
  1. [Local systems](#cosim-nest-lfpy-usecase-installation-on-local-systems)
* [Troubleshooting](#Troubleshooting)
* [Q&A](#Q&A)
---
<!-- TODO cleanup to remove all refs to TVB_NEST_usecase1 -->

<!-- 
# Cosim-NEST-LFPy usecase installation on HPC systems

##### IMPORTANT: In case one of the referenced scripts throws syntax error, it could be that `dash` is being used instead of `bash`. some explanations why could be found on: https://wiki.ubuntu.com/DashAsBinSh

## STEP 1
### Prepare the installation/running location.
creating the installation directory:\
i.e.
 ``` sh
$ cd /p/projects/<jsc.project.name>/<user.name>
$ mkdir <work.dir.name>
$ cd < work.dir.name>
```
e.g.
``` sh
$ cd /p/projects/cslns/slns013
$ mkdir multiscale-cosim
$ cd multiscale-cosim
```

## STEP 2
### Get the Co-Simulation Framework
cloning the TVB-NEST-usecase1 repository along with the required submodules\
``` sh
$ git clone --recurse-submodules --jobs 16 https://github.com/multiscale-cosim/TVB-NEST-usecase1.git
```

## STEP 3
### Set up the runtime environment
executing the installation script in order to set up the run-time environment\
i.e.
``` sh
$ sh ./TVB-NEST-usecase1/installation/hpc/bootstrap_hpc.sh
```

## STEP 4 (OPTIONAL)
### Testing the installation 
executing short tests which import TVB and NEST python packages
#### 4.1. Loading HPC modules and setting CO_SIM_* variables
``` sh
$ source ./TVB-NEST-usecase1/installation/tests/co_sim_vars.source
```

#### 4.2. TVB testing
``` sh
$ python3 ./TVB-NEST-usecase1/installation/tests/tvb_test.py
```
Expected output:
``` sh
TVB on Python OKAY!
```

#### 4.3. NEST testing
``` sh
$ python3 ./TVB-NEST-usecase1/installation/tests/nest_test.py
```
Expected output:
``` sh
...            SimulationManager::run [Info]:
    Simulation finished.
NEST on Python OKAY!
```

---

-->

# Cosim-NEST-LFPy usecase-installation-on-local-systems

## Step 1 Prepare the virtual machine environment

### 1.1 Download and install Vagrant and Virtualbox
- Install [Vagrant version 2.2.19](https://www.vagrantup.com/) or higher
- Install [Virtualbox version 6.1](https://www.virtualbox.org/) or higher

### 1.2 Create directory
e.g.
``` sh
mkdir -p /home/<user.name>/cosim/vagrant
cd /home/<user.name>/cosim/vagrant
```

## Step 2 VM Configuration and installation
### 2.1 Create setup and installation scripts
Go to [installation/local](https://github.com/mfahdaz/Cosim-LFPy/tree/main/installation/local) directory and follow the following steps:

  1. To setup and manage the VM, [Vagrantfile](https://github.com/mfahdaz/Cosim-LFPy/tree/main/installation/local/vagrantfile.file).
<details>
  <summary>(<i>click to expand</i>) </summary>
  
  ``` sh
    # -*- mode: ruby -*-
    # vi: set ft=ruby :

    # All Vagrant configuration is done below. The "2" in Vagrant.configure
    # configures the configuration version (we support older styles for
    # backwards compatibility). Please don't change it unless you know what
    # you're doing.
    Vagrant.configure("2") do |config|
      # The most common configuration options are documented and commented below.
      # For a complete reference, please see the online documentation at
      # https://docs.vagrantup.com.

      # Every Vagrant development environment requires a box. You can search for
      # boxes at https://vagrantcloud.com/search.
      config.vm.box = "ubuntu/focal64"

      # vagrant ouput name on console (during installation)
      config.vm.define "cosim_ubuntu_vm"
      
      # Disable automatic box update checking. If you disable this, then
      # boxes will only be checked for updates when the user runs
      # `vagrant box outdated`. This is not recommended.
      # config.vm.box_check_update = false

      # Create a forwarded port mapping which allows access to a specific port
      # within the machine from a port on the host machine. In the example below,
      # accessing "localhost:8080" will access port 80 on the guest machine.
      # NOTE: This will enable public access to the opened port
      # config.vm.network "forwarded_port", guest: 80, host: 8080

      # Create a forwarded port mapping which allows access to a specific port
      # within the machine from a port on the host machine and only allow access
      # via 127.0.0.1 to disable public access
      # config.vm.network "forwarded_port", guest: 80, host: 8080, host_ip: "127.0.0.1"

      # Create a private network, which allows host-only access to the machine
      # using a specific IP.
      # config.vm.network "private_network", ip: "192.168.33.10"

      # Create a public network, which generally matched to bridged network.
      # Bridged networks make the machine appear as another physical device on
      # your network.
      # config.vm.network "public_network"

      # Share an additional folder to the guest VM. The first argument is
      # the path on the host to the actual folder. The second argument is
      # the path on the guest to mount the folder. And the optional third
      # argument is a set of non-required options.
      config.vm.synced_folder "./shared", "/home/vagrant/shared_data"

      # Provider-specific configuration so you can fine-tune various
      # backing providers for Vagrant. These expose provider-specific options.
      # Example for VirtualBox:
      #
      config.vm.provider "virtualbox" do |vb|
        # Display the VirtualBox GUI when booting the machine
        # vb.gui = true
        # name of the VirtualBox GUI
        vb.name = "cosim_ubuntu_cosim_lfpy"
      
        # Customize the amount of memory on the VM:
        vb.memory = "8192"
        
        #number of cpus
        vb.cpus = "8"

        # vb.customize ["modifyvm", :id, "--uart1", "0x3F8", "4"]
        # vb.customize ["modifyvm", :id, "--uartmode1", "file", File::NULL]
      end
      #
      # View the documentation for the provider you are using for more
      # information on available options.

      # Enable provisioning with a shell script. Additional provisioners such as
      # Ansible, Chef, Docker, Puppet and Salt are also available. Please see the
      # documentation for more information about their specific syntax and use.
      # config.vm.provision "shell", inline: <<-SHELL
      #   apt-get update
      #   apt-get install -y apache2
      # SHELL
      config.vm.provision "shell", path: "bootstrap.sh"
    end
  ```
 </details>
 
   2. To install the required packages and preparing the environment, run [bootstrap.sh](https://github.com/mfahdaz/Cosim-LFPy/tree/main/installation/local/bootstrap.sh) script.

   **NOTE** The bootstrap script is tested on _Ubuntu 22.04.1 LTS (Jammy Jellyfish)_ and _Ubuntu 20.04.5 LTS (Focal Fossa)_.

 <details>
  <summary>(<i>click to expand</i>)</summary>
  
  ``` sh
    #!/bin/bash

#
# USAGE: 
#   a) using defaults <BASELINEPATH>=${HOME} <GITUSERNAME>=multiscale-cosim
#           sh ./TVB_NEST-usecase1_ubuntu_setting_up.sh
#    
#   b) specifiying the parameters   
#        sh ./TVB_NEST_usecase1_ubuntu_setting_up.sh <BASELINEPATH> <GITUSERNAME>
#       e.g. 
#           ./TVB_NEST_usace1_ubuntu_setting_up.sh /opt/MY_COSIM sontheimer

BASELINE_PATH="/home/vagrant"
BASELINE_PATH=${1:-${BASELINE_PATH}}

GIT_DEFAULT_NAME='multiscale-cosim'
GIT_DEFAULT_NAME=${2:-${GIT_DEFAULT_NAME}}

#
# STEP 1 - setting up folder locations
#

[ -d ${BASELINE_PATH} ] \
	|| (echo "${BASELINE_PATH} does not exists"; exit 1;)

#
# Full base path where installation happends:
#
# CO_SIM_ROOT_PATH = /home/<user>/multiscale-cosim/
# or
# CO_SIM_ROOT_PATH = /home/<user>/<git_account_name>/
#
CO_SIM_ROOT_PATH=${BASELINE_PATH}/${GIT_DEFAULT_NAME}

mkdir -p ${CO_SIM_ROOT_PATH}
cd ${CO_SIM_ROOT_PATH}

# CO_SIM_REPOS=${CO_SIM_ROOT_PATH}/cosim-repos
CO_SIM_SITE_PACKAGES=${CO_SIM_ROOT_PATH}/site-packages
CO_SIM_NEST_BUILD=${CO_SIM_ROOT_PATH}/nest-build
CO_SIM_NEST=${CO_SIM_ROOT_PATH}/nest

#
# STEP 2 - installing linux packages
#
# STEP 2.1 - base packages
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip
#
# STEP 2.2 - packages used by NEST, TVB and the use-case per se
sudo apt install -y doxygen
sudo apt install -y libboost-all-dev libgsl-dev libltdl-dev \
                    libncurses-dev libreadline-dev 
sudo apt install -y mpich

#
# STEP 2.3 - switching the default MPI installed packages to MPICH
#   Selection    Path                     Priority   Status
#------------------------------------------------------------
#* 0            /usr/bin/mpirun.openmpi   50        auto mode
#  1            /usr/bin/mpirun.mpich     40        manual mode
#  2            /usr/bin/mpirun.openmpi   50        manual mode
echo "1" | sudo update-alternatives --config mpi 1>/dev/null 2>&1 # --> choosing mpich
echo "1" | sudo update-alternatives --config mpirun 1>/dev/null 2>&1 # --> choosing mpirun
 
#
# STEP 3 - install python packages for the TVB-NEST use-case
#
#
# STEP 4 - TVB
#
# NOTE: Specific versions are required for some packages
#pip install cython elephant mpi4py numpy==1.23.4 pyzmq requests testresources \
#	pandas xarray
pip install --no-cache --target=${CO_SIM_SITE_PACKAGES} lfpykernels mpi4py \
        elephant flask flask-cors gunicorn
	# elephant needed as remnant from other usecases, to be removed
    
    # LFPy specific packages
		 
		
# jupyter notebook stuff
pip install jupyter markupsafe==2.0.1
export PATH=/home/vagrant/.local/bin:$PATH

# 
# STEP 5 - cloning github repos
#
git clone --recurse-submodules --depth 1 --jobs 4 https://github.com/${GIT_DEFAULT_NAME}/Cosim-LFPy.git

#
# STEP 6 - NEST compilation
# International Neuroinformatics Coordinating Facility (INCF) 
# https://github.com/INCF/MUSIC
# https://github.com/INCF/libneurosim

# Cython
export PATH=${CO_SIM_SITE_PACKAGES}/bin:${PATH}
export PYTHONPATH=${CO_SIM_SITE_PACKAGES}:${PYTHONPATH:+:$PYTHONPATH}

mkdir -p ${CO_SIM_NEST_BUILD}
mkdir -p ${CO_SIM_NEST}

cd ${CO_SIM_NEST_BUILD}
cmake \
    -DCMAKE_INSTALL_PREFIX:PATH=${CO_SIM_NEST} \
    ${CO_SIM_ROOT_PATH}/Cosim-LFPy/nest-simulator/ \
    -Dwith-mpi=ON \
    -Dwith-openmp=ON \
    -Dwith-readline=ON \
    -Dwith-ltdl=ON \
    -Dcythonize-pynest=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so

make -j 3
make install
cd ${CO_SIM_ROOT_PATH}

#
# STEP 7 - WORK-AROUNDs (just in case)
#
# removing typing.py as work-around for pylab on run-time
rm -f ${CO_SIM_SITE_PACKAGES}/typing.py
#
# proper versions to be used by TVB
# removing (force) the installed versions 
# __? rm -Rf ${CO_SIM_SITE_PACKAGES}/numpy
# __? rm -Rf ${CO_SIM_SITE_PACKAGES}/gdist
# __? pip install --target=${CO_SIM_SITE_PACKAGES} --upgrade --no-deps --force-reinstall --no-cache matplotlib numpy==1.21
# __? pip install --target=${CO_SIM_SITE_PACKAGES} --upgrade --no-deps --force-reinstall gdist==1.0.2

# even though numpy==1.21 coud have been installed,
# other version could be still present and used

#
# STEP 8 - Generating the .source file based on ENV variables
#
NEST_PYTHON_PREFIX=`find ${CO_SIM_NEST} -name site-packages`
CO_SIM_USE_CASE_ROOT_PATH=${CO_SIM_ROOT_PATH}/Cosim-LFPy
CO_SIM_MODULES_ROOT_PATH=${CO_SIM_ROOT_PATH}/Cosim-LFPy

SUFFIX_PYTHONPATH="\${PYTHONPATH:+:\$PYTHONPATH}"

cat <<.EOSF > ${CO_SIM_ROOT_PATH}/Cosim-LFPy.source
#!/bin/bash
export CO_SIM_ROOT_PATH=${CO_SIM_ROOT_PATH}
export CO_SIM_USE_CASE_ROOT_PATH=${CO_SIM_USE_CASE_ROOT_PATH}
export CO_SIM_MODULES_ROOT_PATH=${CO_SIM_MODULES_ROOT_PATH}
export PYTHONPATH=${CO_SIM_MODULES_ROOT_PATH}:${CO_SIM_SITE_PACKAGES}:${NEST_PYTHON_PREFIX}${SUFFIX_PYTHONPATH}
export PATH=${CO_SIM_NEST}/bin:${PATH}
.EOSF

# 
# STEP 9 - Generating the run_on_local.sh  
cat <<.EORF > ${CO_SIM_ROOT_PATH}/run_on_local.sh
# checking for already set CO_SIM_* env variables
CO_SIM_ROOT_PATH=\${CO_SIM_ROOT_PATH:-${CO_SIM_ROOT_PATH}}
CO_SIM_USE_CASE_ROOT_PATH=\${CO_SIM_USE_CASE_ROOT_PATH:-${CO_SIM_USE_CASE_ROOT_PATH}}
CO_SIM_MODULES_ROOT_PATH=\${CO_SIM_MODULES_ROOT_PATH:-${CO_SIM_MODULES_ROOT_PATH}}
# exporting CO_SIM_* env variables either case
export CO_SIM_ROOT_PATH=\${CO_SIM_ROOT_PATH}
export CO_SIM_USE_CASE_ROOT_PATH=\${CO_SIM_USE_CASE_ROOT_PATH}
export CO_SIM_MODULES_ROOT_PATH=\${CO_SIM_MODULES_ROOT_PATH}
# CO_SIM_ site-packages for PYTHONPATH
export CO_SIM_PYTHONPATH=${CO_SIM_MODULES_ROOT_PATH}:${CO_SIM_SITE_PACKAGES}:${NEST_PYTHON_PREFIX}
# adding EBRAIN_*, site-packages to PYTHONPATH (if needed)
PYTHONPATH=\${PYTHONPATH:-\$CO_SIM_PYTHONPATH}
echo \$PYTHONPATH | grep ${CO_SIM_SITE_PACKAGES} 1>/dev/null 2>&1
[ \$? -eq 0 ] || PYTHONPATH=\${CO_SIM_PYTHONPATH}:\$PYTHONPATH
export PYTHONPATH=\${PYTHONPATH}
# making nest binary reachable
# __ric__? PATH=\${PATH:-$CO_SIM_NEST/bin}
echo \$PATH | grep ${CO_SIM_NEST}/bin 1>/dev/null 2>&1
[ \$? -eq 0 ] || export PATH=$CO_SIM_NEST/bin:\${PATH}
python3 \${CO_SIM_USE_CASE_ROOT_PATH}/main.py \\
    --global-settings \${CO_SIM_MODULES_ROOT_PATH}/EBRAINS_WorkflowConfigurations/general/global_settings.xml \\
    --action-plan \${CO_SIM_USE_CASE_ROOT_PATH}/userland/configs/local/plans/cosim_nest_lfpy_potjans_2014_local.xml
.EORF

cat <<.EOKF >${CO_SIM_ROOT_PATH}/kill_co_sim_PIDs.sh
for co_sim_PID in \`ps aux | grep Cosim-LFPy | sed 's/user//g' | sed 's/^ *//g' | cut -d" " -f 1\`; do kill -9 \$co_sim_PID; done
.EOKF

#
# STEP 10 - THIS IS THE END!
#
echo "SETUP DONE!"
```
</details>

Important: 
- Ensure exact filenames `Vagrantfile` and `bootstrap.sh`.
- Create a folder to share data between VM and physical OS (see line 49 in Vagrantfile).
``` sh
 mkdir shared
```

### 2.2 Start the virtual machine and installation process
Run the following command from within the newly created directory.
``` sh
vagrant up
```
On first startup, the command will run the bootstrap script. The installation process will take several minutes.

After installation is complete, the VM can be accessed. 
``` sh
vagrant ssh
```
The VM can be stopped (after exiting) by running
``` sh
vagrant halt
```


---

# TROUBLESHOOTING
WIP
---

# Q&A
WIP
