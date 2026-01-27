#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=im_reasoning_test 
#SBATCH --time=1:00:00 
#SBATCH --ntasks=1 
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=3 
#SBATCH --mem=64gb 
#SBATCH --partition=short 

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

export JOB_HF_HOME="/mnt/evafs/groups/jrafalko-lab/huggingface/tmp_${SLURM_JOB_ID}"
mkdir -p ${JOB_HF_HOME}

export JOB_TMPDIR="/mnt/evafs/groups/jrafalko-lab/tmp_${SLURM_JOB_ID}"
mkdir -p ${JOB_TMPDIR}

cd /mnt/evafs/groups/jrafalko-lab

source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

cd inzynierka

python -m src.tests.evaluation_test

rm -rf ${JOB_HF_HOME}
rm -rf ${JOB_TMPDIR}