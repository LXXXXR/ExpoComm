# ExpoComm: Cluster Setup and Reproduction Guide

This guide describes how to install the ExpoComm environment and reproduce the
provided experiments on a Slurm-based GPU cluster.

## Prerequisites

Before starting, make sure you have:

- Access to a Slurm cluster with a GPU partition.
- A Linux login node with Git and `wget` available.
- A Weights & Biases account (optional, but recommended for viewing training
  metrics).

## 1. Install Miniconda

Run the following commands on the login node if Conda is not already
available:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Accept Conda's Terms of Service when prompted. If needed, they can also be
accepted explicitly:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 2. Clone the Repository

```bash
git clone <REPOSITORY_URL> ExpoComm
cd ExpoComm
```

All remaining commands in this guide should be run from the repository root.


## 3. Create the Conda Environment

Submit the setup job:

```bash
sbatch setup_expocomm_test.sh
```

The job creates the Conda environment, installs the Python dependencies, and
copies ExpoComm's customised MAgent scenarios into PettingZoo.

Monitor its status and logs with:

```bash
squeue -u <USERNAME>
squeue -j <JOB_ID>
tail -f setup_expocomm_<JOB_ID>.out
```

Wait until the setup job has completed successfully before submitting an
experiment. The `tail -f` command can be stopped with `Ctrl+C`.

## 4. Sign In to Weights & Biases

Activate the environment and authenticate with the API key from your W&B
account:

```bash
conda activate ExpoComm_test
wandb login
```

Training can run without using the web dashboard, but W&B is useful for
monitoring metrics and comparing runs.

## 5. Submit an Experiment

The reproduction script takes three arguments:

```bash
sbatch run_reproduce_expocomm.sh <ALGORITHM> <ENVIRONMENT> <EXPERIMENT>
```

The names correspond to configuration files under `src/config/algs`,
`src/config/envs`, and `src/config/exp`, respectively. Refer to the paper for
the experiment settings and interpretations.

### Adversarial Pursuit

```bash
# 25 agents: K = 1 and K = ceil(log2 N)
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n6 MAgent_AdvPursuit ExpoComm_AdvPursuit45_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n6   MAgent_AdvPursuit ExpoComm_AdvPursuit45_s0

# 45 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_AdvPursuit ExpoComm_AdvPursuit60_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_AdvPursuit ExpoComm_AdvPursuit60_s0

# 61 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_AdvPursuit ExpoComm_AdvPursuit70_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_AdvPursuit ExpoComm_AdvPursuit70_s0
```

### Battle

```bash
# 20 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n6 MAgent_Battle ExpoComm_Battle25_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n6   MAgent_Battle ExpoComm_Battle25_s0

# 42 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle ExpoComm_Battle35_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_Battle ExpoComm_Battle35_s0

# 64 agents
sbatch -p gpu-long --nodelist=xgpf7 run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle ExpoComm_Battle40_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_Battle ExpoComm_Battle40_s0
```

### Choose a GPU Node (Optional)

By default, `run_reproduce_expocomm.sh` requests one GPU and pins the job to
`xgpf10`:

```bash
#SBATCH --nodelist=xgpf10
```

If another GPU node is available, replace `xgpf10` with that node's name. To
let Slurm select any eligible GPU node automatically, remove or comment out
the `#SBATCH --nodelist=xgpf10` line. You may also need to set
cluster-specific options such as `--partition` or `--account`.

Alternatively, request a node for one submission without changing the script:

```bash
sbatch --nodelist=<GPU_NODE> run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle ExpoComm_Battle35_s0
```

## Troubleshooting

- **`conda: command not found`:** set `CONDA_BASE` to the Miniconda or Anaconda
  installation before submitting the setup job.
- **The training job cannot activate Conda:** update the `source .../conda.sh`
  path in `run_reproduce_expocomm.sh` to your installation.
- **The requested node is unavailable:** remove or replace the hard-coded
  `--nodelist` directive in the reproduction script.
- **Setup cannot find the repository:** ensure the setup script's `cd ~/ExpoComm`
  line matches the location where you cloned this repository.
