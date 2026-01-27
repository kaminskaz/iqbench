#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=im_reasoning_test 
#SBATCH --time=5:00:00 
#SBATCH --ntasks=1 
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4 
#SBATCH --mem=64gb 
#SBATCH --partition=short
 
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

cd inzynierka

python -m src.tests.model_supportability_test
