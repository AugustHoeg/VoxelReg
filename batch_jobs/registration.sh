#!/bin/bash

# Configuration
# This is what you should change for your setup
VENV_NAME=venv         # Name of your virtualenv (default: venv)
VENV_DIR=.             # Where to store your virtualenv (default: current directory)
PYTHON_VERSION=3.11.9  # Python version (default: 3.9.14)
CUDA_VERSION=12.1      # CUDA version (default: 11.6)

#BSUB -q hpc
#BSUB -J VoxelReg
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
###BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# request 40GB of system-memory rusage=40
###BSUB -R "select[gpu40gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -u "august.hoeg@gmail.com"
#BSUB -B
#BSUB -N
#BSUB -oo batch_outputs/output_august_%J.out
#BSUB -eo batch_errors/error_august_%J.out

# Exits if any errors occur at any point (non-zero exit code)
set -e

# Load modules
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "matplotlib/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "pandas/")
#module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "cv2/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "os/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "glob/")
module load cuda/$CUDA_VERSION
CUDNN_MOD=$(module avail -o modulepath -t cudnn | grep "cuda-${CUDA_VERSION}" 2>&1 | sort | tail -n1)
if [[ ${CUDNN_MOD} ]]
then
    module load ${CUDNN_MOD}
fi

# Create virtualenv if needed and activate it
if [ ! -d "${VENV_DIR}/${VENV_NAME}" ]
then
    echo INFO: Did not find virtualenv. Creating...
    virtualenv "${VENV_DIR}/${VENV_NAME}"
fi
source "${VENV_DIR}/${VENV_NAME}/bin/activate"

echo "About to run scripts"

SAMPLE_PATH=""
MOVING_PATH="Larch_A_LFOV_crop_full_height.npy"
FIXED_PATH="Larch_A_4x_pos1_down_4.npy"
OUT_NAME="Larch_A_LFOV_pos1_registered"

python -u elastix_automatic_registration.py --sample_path "$SAMPLE_PATH" \
--fixed_path "$FIXED_PATH" --moving_path "$MOVING_PATH" --out_name "$OUT_NAME" --run_type "DTU_HPC" \
--center 1450 161 161 --size 1 1 1 --spacing 25 20 20

echo "Done"

#python -u preprocess.py --sample_path "" --scan_path --out_path --run_type "DTU_HPC" --min_size --max_size --margin_percent --divis_factor --save_downscaled False --f 4 --mask_threshold "otsu"


#python -u elastix_automatic_registration.py --sample_path "" --scan_path --out_path --run_type "DTU_HPC" --min_size --max_size --margin_percent --divis_factor --save_downscaled False --f 4 --mask_threshold "otsu"
