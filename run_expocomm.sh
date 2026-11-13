#!/bin/bash
#SBATCH --job-name=expocomm
#SBATCH --output=slurm/expocomm_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --nodelist=xgpf10


set -e
cd ~/ExpoComm

echo "Node: $(hostname)  Arch: $(uname -m)  Date: $(date)"
if [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: node không phải x86_64, env ExpoComm không chạy được."
    exit 1
fi
nvidia-smi -L
free -h

source /home/b/bach66/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/scratch/b/bach66/conda/envs/ExpoComm



( while true; do echo "[mem $(date +%T)] $(free -m | awk '/Mem:/{print $3" MB used"}')"; sleep 60; done ) &
MON=$!

python src/main.py \
    --config=iql_ExpoComm_one_peer_n6 \
    --env-config=MAgent_AdvPursuit \
    --exp-config=ExpoComm_AdvPursuit45_s0

kill $MON 2>/dev/null || true