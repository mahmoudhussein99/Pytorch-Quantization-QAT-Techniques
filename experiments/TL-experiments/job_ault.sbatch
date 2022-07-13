#!/bin/bash -l
#
#SBATCH --job-name=cifar10_Training
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=amdv100,intelv100,amdrtx,amda100
#SBATCH --constraint=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --account=mhussein

cmd=$1

export PYTHONPATH="${PYTHONPATH}:/users/mhussein/State-of-Quantization-in-DL"
source ../../env/bin/activate
echo $cmd
srun $cmd