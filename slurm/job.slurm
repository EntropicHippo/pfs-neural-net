#!/bin/bash
#SBATCH --job-name=BipartiteMPNN    # create a short name for your job
#SBATCH --output=slurm_out.txt      # rename output file
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                    # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4                # number of gpus per node
#SBATCH --constraint=gpu80          # explicitly run on A100 GPUs
#SBATCH --time=01:00:00             # total run time limit (HH:MM:SS)

cd ..
module purge
module load anaconda3/2024.10
conda activate torch-gnn

cd src
python train.py
