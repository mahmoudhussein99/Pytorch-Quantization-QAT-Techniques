#!/bin/bash -l
#
#SBATCH --job-name=install-setup.py
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=amdv100,intelv100,amdrtx,amda100
#SBATCH --constraint=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --account=mhussein



source env/bin/activate

srun python3 setup.py gpu install 
