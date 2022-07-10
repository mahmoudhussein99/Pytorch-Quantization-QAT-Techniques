#!/bin/bash -l
#
#SBATCH --job-name=ImageNet_SpeedUp
#SBATCH --time=10:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_imageNet-%j.out
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=g34

module load daint-gpu
source /scratch/snx3000/sashkboo/State-of-Quantization-in-DL/env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/scratch/snx3000/sashkboo/State-of-Quantization-in-DL"

export OMPI_COMM_WORLD_SIZE=$((SLURM_NTASKS * SLURM_NTASKS_PER_NODE))
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ipogif0


echo "-----------------------------"
echo "World size: ${SLURM_NTASKS}"
echo "-----------------------------"

srun python3 imagenet_slurm.py /project/g34/imagenet --batch-size 64 -a resnet50  --epochs 90 --find_unused_param True  --resume checkpoints/eternal-leaf-2335 \
             --dist-backend nccl --multiprocessing-distributed --workers 8 --pact_reg 0.00001 \
             --act_qmode pact --act_bits 4 --weight_qmode sawb --weight_bits 4   \
            --error_qmode absmax --error_rounding stochastic --error_man 0 --error_sig 3 --error_rep rdx2 \
            --first_weight_qmode lsq_weight --first_weight_bits 4 --first_error_qmode adaptive --first_error_sig 3 --first_error_man 0 --first_error_rep rdx2 --first_error_rounding stochastic      \
            --first_act_qmode pact --first_act_bits 8
