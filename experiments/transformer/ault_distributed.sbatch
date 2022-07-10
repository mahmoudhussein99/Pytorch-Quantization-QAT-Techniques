#!/bin/bash -l
#
#SBATCH --job-name=big_tformer_Training
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=4
#SBATCH --mem=128G
#SBATCH --partition=amdrtx
#SBATCH --constraint=gpu
#SBATCH --output=slurm_distributed-%j.out
#SBATCH --account=sashkboo



export PYTHONPATH="${PYTHONPATH}:/users/sashkboo/State-of-Quantization-in-DL"
source ../../env/bin/activate


data_dir=fairseq/data-bin/wmt14_en_de/
SAVEDIR="checkpoints/baseline2"

PORT=9218
srun fairseq-train $data_dir \
    --distributed-port 42341 \
    --distributed-backend nccl \
    --arch tiny_transformer_v2 \
    --user-dir q_module \
    --task qtranslation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --stop-min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --weight-decay 0.0 \
    --max-tokens 4096 \
    --save-dir ${SAVEDIR} \
    --update-freq 1 \
    --no-progress-bar \
    --log-format simple \
    --log-interval 100 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \