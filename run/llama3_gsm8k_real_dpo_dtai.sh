#!/bin/bash
#SBATCH -N 1             
#SBATCH -t 1:00:00
#SBATCH --partition=ghx4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-node 4
#SBATCH -J llama3_gsm8k_dpo
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Current directory: $(pwd)"
echo "Job started at $(date)"
date_str=$(date '+%S_%d_%b_%Y')
echo "Running job on $(hostname)"
echo "Job started at $(date)"

source ~/.bashrc
mkdir -p slurm
prefix="llama3_gsm8k_dpo"
time srun --label apptainer exec --nv \
  --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
  --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
  --bind $HOME:$HOME \
  --bind $MY_PROJECT:$MY_PROJECT \
  $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
  "source ~/.bashrc && bash run/llama3_gsm8k_real_dpo.sh" \
  > slurm/${prefix}_${date_str}_${SLURM_JOB_ID}.log 2>&1 

echo "Job finished at $(date)"

# apptainer exec --nv \
#   --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
#   --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
#   --bind $HOME:$HOME \
#   --bind $MY_PROJECT:$MY_PROJECT \
#   $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
#   "source ~/.bashrc && bash run/llama3_gsm8k_real_dpo.sh"

 

# apptainer exec --nv \
#   --env PYTHONPATH="$HOME/.my_envs/llamafactory" \
#   --env PYTHONUSERBASE="$HOME/.my_envs/llamafactory" \
#   --bind $HOME:$HOME \
#   --bind $MY_PROJECT:$MY_PROJECT \
#   $CONTAINER_DIR/pytorch_24.09-py3.sif bash -c \
#   "source ~/.bashrc && bash"

# grep -hE "^(Name|Version):" $(find $HOME/.my_envs/llamafactory/lib -name "METADATA" -o -name "PKG-INFO") | paste - - | awk '{print $2"=="$4}' > requirements_my_env.txt
