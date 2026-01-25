#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=im_reasoning_test 
#SBATCH --time=5:00:00 
#SBATCH --ntasks=1 
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=3 
#SBATCH --mem=64gb 
#SBATCH --partition=short 

DATASET_NAME=${1:-bp}

echo "Dataset: $DATASET_NAME"

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

export JOB_HF_HOME="/mnt/evafs/groups/jrafalko-lab/huggingface_${SLURM_JOB_ID}"
mkdir -p ${JOB_HF_HOME}

export JOB_TMPDIR="/mnt/evafs/groups/jrafalko-lab/tmp_${SLURM_JOB_ID}"
mkdir -p ${JOB_TMPDIR}

source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

python -m src.tests.ensemble_test \
    --dataset_name "$DATASET_NAME"

rm -rf ${JOB_HF_HOME}
rm -rf ${JOB_TMPDIR}