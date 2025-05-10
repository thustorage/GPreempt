# [USENIX ATC'25 Artifact] GPreempt: GPU Preemptive Scheduling Made General and Efficient

Welcome to the artifact repository of USENIX ATC'25 accepted paper: *GPreempt: GPU Preemptive Scheduling Made General and Efficient*.

Should there be any questions, please contact the authors in HotCRP. The authors will respond to each question within 24hrs and as soon as possible.

## Environment Setup

**To artifact reviewers:** please skip this section and go to "[Evaluate the Artifact](#evaluate-the-Artifact)". This is because we have already set up the required environment on the provided platform. The following instructions are for users who wish to set up GPreempt from scratch on their own machines.

### Prerequisites

- Ubuntu 22.04
- NVIDIA driver 550.120
- Install modified NVIDIA kernel module
- GDRCopy

### Compile & Install the NVIDIA kernel module

**⚠️ Disclaimer**

This project contains **modifications to the official NVIDIA kernel driver**, and is intended **for research purposes only**. This is the most critical and potentially complex step if setting up a new environment. Please follow the instructions carefully.

**Please note:**

- Use of this driver may cause system instability, hardware malfunction, or void official NVIDIA support.
- **DO NOT use this in production environments. Use it at your own risk.**

The modified NVIDIA kernel module is based on the official NVIDIA driver version 550.120. We provide a patch file to apply the modifications. The patch file is located in the `patch` directory of this repository.

`${REPO_ROOT}` refers to the root directory of this GPreempt artifact repository.

```shell
# Clone the official NVIDIA open-gpu-kernel-modules repository
git clone https://github.com/NVIDIA/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
git checkout 550.120 # Ensure you are on the correct base version

# Apply the patch from the GPreempt repository
# Make sure ${REPO_ROOT} is correctly set to the path of the GPreempt artifact
# Alternatively, copy the patch file into the current directory:
# cp ${REPO_ROOT}/patch/driver.patch .
# git apply driver.patch
git apply ${REPO_ROOT}/patch/driver.patch
```

Then, you need to build and install the modified NVIDIA kernel module. The following commands will do this:
```shell
make modules -j$(nproc)
sudo make modules_install -j$(nproc)
sudo depmod
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod gdrcopy
sudo rmmod nvidia
```

### Install GDRCopy

Please see the official GDRCopy repository for installation instructions: [GDRCopy](https://github.com/NVIDIA/gdrcopy)

### How to compile models

Use the python scripts in `scripts/compile` folder. `compile-bert.py` is used to compile bert model. Others use `compile.py`

A self-modified version of tvm is needed. So before running the script, set the following environment variable:

```shell
export PYTHONPATH=/path/to/tvm/python:$PYTHONPATH
pip install -r requirements.txt
```

Transform and compile the model:

```shell
cd model
make
```

### Build GPreempt
Remember to run `git submodule update --init --recursive` before building GPreempt. This will download the required submodules.

```shell
cd ${REPO_ROOT}
mkdir build
cmake ..
make -j$(nproc)
```

## Evaluate the Artifact

### Hello-world example

To verify that everything is prepared, you can run a hello-world example that verifies GPreempt's functionality, please run the following command:

```shell
scripts/ae/hello_world.sh
```

It will run for a few seconds and, on success, output something like below:

```shell
PASS set priority
PASS GDRcopy
PASS all
```

Please file an issue if this does not work. Thank you very much.

### Run all experiments

There is an all-in-one AE script for your convenience:

```shell
scripts/ae/run_all.sh <path_to_result_dir>
```

where `<path_to_result_dir>` is the directory where you want to store the results. For example, if you want to store the results in `result-reviewer-A`, please run:
```shell
scripts/ae/run_all.sh result-reviewer-A
```

By default, if you do not specify the path, it will store the results in `results` folder.

This script will run for approximately 2 hours.

Since the original REEF repository only supports the ROCm driver of Ubuntu 18.04, the corresponding code cannot be executed on the current version of Ubuntu. Therefore, we provide two options:
- **(Default)**: Use Pre-computed Results for REEF/AMD:
The run_all.sh script, by default, will copy our pre-computed REEF/AMD experiment logs into your <path_to_result_dir>. The plotting scripts will then use these logs to generate the comparative figures. This ensures that reviewers can reproduce the figures shown in the paper without needing a separate AMD environment. You will see messages in the script output indicating when these pre-computed results are being used.
- **(Optional)**: We also installed an old version of Ubuntu system on the provided machine. However, since my teammates need to use that machine for other experiments at the moment, not all time slots are available. Please make an appointment with us on hotcrp. We will restart the server to the corresponding version system for evaluation.

#### Note
- Since the AMD GPU is installed on another server, we remotely connect to that server via SSH to run the corresponding programs. You can refer to the details in `scripts/ae/run-hip.sh`. You can also access the server we provided by running `ssh 10.0.2.190`; the GPreempt code is located at `~/workdir/gpreempt`.

### Plot the figures

#### (Recommended) For Visual Studio Code users

Please install the Jupyter extension in VSCode. Then, please open `scripts/plot/plot.ipynb`.

Please use the Python 3.10.12 (/usr/bin/python) kernel.

Then, you can run each cell from top to bottom. The first cell contains prelude functions and definitions, so please run it first. Each other cell plots a figure or table.

#### For others

Please run the plotter script in the `scripts/ae` directory:

```shell
cd scripts/plot
python3 plot.py -r <path_to_result_dir> -f <figures_you_want_to_draw>
```

where `<path_to_result_dir>` is the directory where you stored the results. For example, if you stored the results in `result-reviewer-A`, please run: 
```shell
cd scripts/plot
python3 plot.py -r result-reviewer-A
```

The command above will plot all figures and tables by default, and the results will be stored in the `<path_to_result_dir>/figures` directory. So, please ensure that you have finished running the all-in-one AE script before running the plotter.

The plotter supports specifying certain figures or tables to plot by command-line arguments. For example:

```shell
python3 plot.py -r ../../result-reviewer-A -f 4a,4b
```

Please refer to python3 plot.py --help or the script's comments for available figure keys.


### Major claims of this paper

The main claims of this paper, which can be verified through the provided artifact, are:

1. **Low Preemption Latency:** GPreempt achieves low preemption-induced latency for latency-critical (LC) tasks when co-located with best-effort (BE) tasks. This is demonstrated by the experimental results presented in Figure 4, Figure 6, and Figure 7.
  
2. **High System Throughput:** GPreempt maintains high system throughput for BE tasks even when co-located with preemptible LC tasks. This is supported by the data in Figure 5.
  
3. **Broad Applicability to Diverse Workloads:** GPreempt effectively supports preemption for a diverse range of GPU applications, including complex, non-idempotent workloads such as graph analytics (e.g., PageRank) and scientific simulations (e.g., weather simulation). The successful execution and performance benefits shown in Figures 4 and 5 (which use such applications) demonstrate this capability.

## Workloads and Benchmarking

### DISB

This project utilizes workloads derived from the **SJTU-IPADS/disb** repository to evaluate its performance.

-   **Original Repository:** [https://github.com/SJTU-IPADS/disb](https://github.com/SJTU-IPADS/disb)
  
-   **Original License:** Apache License, Version 2.0
  

In accordance with the Apache 2.0 License requirements:

1.  **License Copy:** A full copy of the Apache License 2.0, under which the `disb` code is distributed, is included in this repository([LICENSE](./LICENSES/Apache-2.0.txt)).

2.  **Statement of Changes:**  Modifications made to the original `disb` code/scripts used in this project are explicitly noted within the relevant source files. Please refer to those locations for details on the changes.

We gratefully acknowledge the work of the contributors to the `SJTU-IPADS/disb` project for providing these valuable resources.

### Synthetic Workload

#### Synthetic Workload Y (Scientific Computing)

This workload simulates scientific computing tasks, adapted from the **miniWeather** mini-app.

- **Description:** Represents computational kernels typical in weather modeling applications.
  
- **Original Source:** The MiniWeather Mini App by mrnorman
  
- **Original Repository:** [https://github.com/mrnorman/miniWeather](https://github.com/mrnorman/miniWeather)
  
- **Original License:** BSD 2-Clause License
  

**License Compliance:**

1. **License Copy:** A copy of the BSD 2-Clause License, under which the original miniWeather code is distributed, is included in this repository([LICENSE BSD 2-Clause](./LICENSES/BSD-2-Clause.txt)).
  
2. **Statement of Changes:** This project uses code adapted from the original miniWeather. Modifications made for integration and evaluation purposes are noted within the relevant source files in this repository.
  

We acknowledge contributors for the miniWeather application.

#### Synthetic Workload Z (Graph Computing)

This workload represents graph computing tasks, adapted from the **EMOGI** repository.

- **Description:** The original repository provides multiple GPU Graph computing applications including CC, BFS, SSSP, Pagerank.
  
- **Original Repository:** [https://github.com/illinois-impact/EMOGI](https://github.com/illinois-impact/EMOGI)

- **Statement of Changes:** This project uses code adapted from the original EMOGI. Modifications made for integration and evaluation purposes are noted within the relevant source files in this repository.

We acknowledge the authors of the EMOGI repository.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.