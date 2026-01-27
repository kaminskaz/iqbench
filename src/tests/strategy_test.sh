#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=im_reasoning_test 
#SBATCH --time=5:00:00 
#SBATCH --ntasks=1 
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6 
#SBATCH --mem=64gb 
#SBATCH --partition=short 

DATASET_NAME=${1:-bp}
STRATEGY=${2:-direct}
MODEL_NAME=${3:-"OpenGVLab/InternVL3-8B"}
RESTART_PROBLEM_ID=${4:-""}
RESTART_VERSION=${5:-""}
PARAM_SET_NUMBER=${6:-"1"}
PROMPT_NUMBER=${7:-"1"}
FORCE_NEW_VERSION=${8:-"False"}

echo "Dataset: $DATASET_NAME"
echo "Strategy: $STRATEGY"
echo "Model: $MODEL_NAME"
echo "Restart Problem ID: $RESTART_PROBLEM_ID"
echo "Parameter Set Number: $PARAM_SET_NUMBER"
echo "Prompt Number: $PROMPT_NUMBER"
echo "Force New Version: $FORCE_NEW_VERSION"

CURRENT_FORCE="$FORCE_NEW_VERSION"

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=DEBUG

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

cd /mnt/evafs/groups/jrafalko-lab/inzynierka

for i in {1..50}
do
   echo "Starting run $i of 50..."
   
   CMD="python -m src.tests.strategy_test \
      --dataset_name $DATASET_NAME \
      --strategy $STRATEGY \
      --model_name $MODEL_NAME \
      --param_set_number $PARAM_SET_NUMBER \
      --prompt_number $PROMPT_NUMBER"

   if [ "$CURRENT_FORCE" = "True" ]; then
      CMD="$CMD --force_new_version"
   fi

   $CMD
        
   status=$?

   CURRENT_FORCE="False"

   if [ $status -eq 2 ]; then
      echo "Exit code 2 detected. Terminating loop - all problems processed."
      exit 2
   elif [ $status -eq 1 ]; then
      echo "Exit code 1 detected. Terminating loop - critical problem."
      exit 1
   elif [ $status -ne 0 ]; then
      echo "Run $i failed with exit code $status, but continuing to next iteration..."
   fi
done
    