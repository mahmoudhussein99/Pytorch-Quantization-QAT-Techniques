#!/bin/bash -l
#
#SBATCH --job-name=ImageNet_Training
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=128
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --mem=128G
#SBATCH --partition=amdrtx
#SBATCH --constraint=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --account=mhussein

export PYTHONPATH="${PYTHONPATH}:/users/mhussein/State-of-Quantization-in-DL"
source ../../env/bin/activate


echo "-----------------------------"
echo "World size: ${SLURM_NTASKS}"
echo "-----------------------------"


cmd=$1
echo $cmd
srun $cmd
