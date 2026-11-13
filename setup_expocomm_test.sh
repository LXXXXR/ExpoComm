#!/usr/bin/env bash
#SBATCH --job-name=setup_expocomm
#SBATCH --output=setup_expocomm_%j.out
#SBATCH --error=setup_expocomm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Create and populate the Conda environment needed by ExpoComm.
# Submit from the repository root with: sbatch setup_expocomm_test.sh

set -Eeuo pipefail

ENV_NAME="ExpoComm_test"
# Slurm can execute a copied script from a spool directory. Use the directory
# from which sbatch was called so requirements.txt is resolved reliably.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ ! -f "${REPO_DIR}/requirements.txt" ]]; then
    echo "ERROR: ${REPO_DIR} is not the ExpoComm repository (requirements.txt is missing)."
    exit 1
fi

# Slurm jobs use a non-interactive shell, so Conda must be explicitly loaded.
if ! command -v conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: Conda was not found. Export CONDA_BASE=/path/to/miniconda before submitting."
        exit 1
    fi
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
    echo "Creating Conda environment '${ENV_NAME}'."
    conda create -y -n "${ENV_NAME}" python=3.8
fi

conda activate "${ENV_NAME}"
cd "${REPO_DIR}"

python -m pip install "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install --no-build-isolation "gym==0.21.0"
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/moratodpg/imp_marl.git
python -m pip install wandb einops torch_scatter
python -m pip install "magent==0.1.14"
python -m pip install "supersuit==3.3.2"
python -m pip install "pettingzoo==1.12.0"
python -m pip install "numpy==1.23.5"
python -m pip install setproctitle
# Install ExpoComm's customised MAgent scenarios into this environment's
# PettingZoo package. Resolving the location through the active Python avoids
# any machine- or Conda-prefix-specific path.
cd ~/ExpoComm
REPO_DIR=$PWD
PZ_MAGENT_DIR="$(python -c 'import pathlib, pettingzoo; print(pathlib.Path(pettingzoo.__file__).parent / "magent")')"

cp "${REPO_DIR}/env/battle_v3_view7.py" "${PZ_MAGENT_DIR}/"
cp "${REPO_DIR}/env/adversarial_pursuit_view8_v3.py" "${PZ_MAGENT_DIR}/"
ls "${PZ_MAGENT_DIR}" | grep view

echo "Setup complete. Activate it with: conda activate ${ENV_NAME}"
