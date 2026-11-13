#!/usr/bin/env bash
#SBATCH --job-name=expocomm
#SBATCH --output=slurm/expocomm_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --nodelist=xgpf10

# Submit one reproduction run with:
#   sbatch run_reproduce_expocomm.sh <algorithm> <environment> <experiment>
#
# Examples (the paper's MAgent results):
#   sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n6 MAgent_AdvPursuit ExpoComm_AdvPursuit45_s0
#   sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n6   MAgent_Battle      ExpoComm_Battle25_s0
#   sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle      ExpoComm_Battle40_s0
#
# Examples (the paper's IMP results):
#   sbatch run_reproduce_expocomm.sh qmix_ExpoComm_one_peer_n7_cont IMP_struct_c_50_cc ExpoComm_struct_c_50_cc_s0
#   sbatch run_reproduce_expocomm.sh qmix_ExpoComm_static_n8_cont   IMP_struct_uc_100_cc ExpoComm_struct_uc_100_cc_s0

set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch $0 <algorithm> <environment> <experiment>"
    exit 2
fi

ALGORITHM="$1"
ENVIRONMENT="$2"
EXPERIMENT="$3"
CONDA_ENV="${CONDA_ENV:-ExpoComm_test}"
# Slurm may execute a copied batch script from a spool directory.  Its submit
# directory is the reliable project location when the job was submitted there.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ ! -f "${REPO_DIR}/src/main.py" ]]; then
    echo "ERROR: ${REPO_DIR} is not the ExpoComm repository (src/main.py is missing)."
    exit 1
fi

# Match the cluster-specific setup used by run_expocomm.sh.
source /home/b/bach66/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

echo "Node: $(hostname)"
echo "Environment: ${CONDA_ENV}"
echo "Algorithm: ${ALGORITHM}"
echo "Environment config: ${ENVIRONMENT}"
echo "Experiment config: ${EXPERIMENT}"
nvidia-smi -L

python src/main.py \
    --config="${ALGORITHM}" \
    --env-config="${ENVIRONMENT}" \
    --exp-config="${EXPERIMENT}"
