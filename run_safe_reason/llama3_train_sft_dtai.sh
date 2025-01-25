#!/bin/bash
#SBATCH -N 1             
#SBATCH -t 0:30:00
#SBATCH --partition=ghx4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-node 4
#SBATCH -J llama3_sft
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Current directory: $(pwd)"
echo "Job started at $(date)"
date_str=$(date '+%d_%b_%Y')
echo "Running job on $(hostname)"
echo "Job started at $(date)"

source ~/.bashrc
mkdir -p slurm
prefix="llama3_sft"
time srun --label apptainer exec --nv \
  --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
  --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
  --bind $HOME:$HOME \
  --bind $MY_PROJECT:$MY_PROJECT \
  $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
  "source ~/.bashrc && bash run_safe_reason/llama3_train_sft.sh" \
  > slurm/${prefix}_${date_str}_${SLURM_JOB_ID}.log 2>&1 

prefix="llama3_sft_eval"
dataset="sr_sft_preliminary_meta-llama-Meta-Llama-3-8B-Instruct_5_5_50_500_0"
experiment_name=llama3_sft_${dataset}_tbs8_ebs8_gas2_lr5e-6_cl4096
time srun --label apptainer exec --nv \
  --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
  --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
  --bind $HOME:$HOME \
  --bind $MY_PROJECT:$MY_PROJECT \
  $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
  "source ~/.bashrc && bash run_safe_reason/train_eval.sh ${experiment_name}" \
  > slurm/${prefix}_${date_str}_${SLURM_JOB_ID}.log 2>&1

echo "Job finished at $(date)"

# apptainer exec --nv \
#   --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
#   --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
#   --bind $HOME:$HOME \
#   --bind $MY_PROJECT:$MY_PROJECT \
#   $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
#   "source ~/.bashrc && bash run_safe_reason/llama3_train_sft.sh"

# apptainer exec --nv \
#   --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
#   --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
#   --bind $HOME:$HOME \
#   --bind $MY_PROJECT:$MY_PROJECT \
#   $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
#   "source ~/.bashrc && bash"

# grep -hE "^(Name|Version):" $(find $HOME/.my_envs/llamafactory/lib -name "METADATA" -o -name "PKG-INFO") | paste - - | awk '{print $2"=="$4}' > requirements_my_env.txt
