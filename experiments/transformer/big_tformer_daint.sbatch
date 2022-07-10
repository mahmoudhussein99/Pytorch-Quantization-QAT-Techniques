#!/bin/bash -l
#
#SBATCH --job-name=big_tformer_speedup
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%j.out
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --account=g34

module load daint-gpu
source /scratch/snx3000/sashkboo/State-of-Quantization-in-DL/env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/scratch/snx3000/sashkboo/State-of-Quantization-in-DL"



seed=2
data_dir=fairseq/data-bin/wmt14_en_de/
SAVEDIR="checkpoints/big_base_qtformer_speedUp"


srun fairseq-train $data_dir \
    --distributed-port 42341 \
    --distributed-backend nccl \
    --arch qtransformer_vaswani_wmt_en_de_big \
    --user-dir q_module \
    --task qtranslation \
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
    --max-tokens 3000 \
    --save-dir ${SAVEDIR} \
    --update-freq 1 \
    --no-progress-bar \
    --log-format simple \
    --log-interval 100 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \
    --share-decoder-input-output-embed \
    --wandb-project state_quantization_neurips22 \
    --max-epoch 32 \
    --optimizer adam \
      --ddp-backend=no_c10d \
       --fc_act_qmode dorefa_act --fc_act_bits 4 --fc_weight_qmode lsq_weight --fc_weight_bits 4 \
      --fc_error_qmode adaptive --fc_error_rep rdx2 --fc_error_rounding stochastic --fc_error_sig 3 --fc_error_man 0 \
      --mha_linear_act_qmode lsq_act --mha_linear_act_bits 8  --mha_linear_weight_qmode lsq_weight --mha_linear_weight_bits 8 \
      --mha_linear_error_qmode absmax --mha_linear_error_rep fp --mha_linear_error_rounding nearest --mha_linear_error_sig 5 --mha_linear_error_man 2 \
      --last_weight_qmode lsq_weight --last_weight_bits 8 --last_error_qmode adaptive --last_error_rep fp  --last_error_rounding nearest  --last_error_sig 5  --last_error_man 2 \
      --last_act_qmode lsq_act --last_act_bits 8 \
      --mha_act_qmode pact --mha_act_bits 8 \
      --mha_error_qmode adaptive --mha_error_rep fp  --mha_error_rounding nearest  --mha_error_sig 5  --mha_error_man 2 \
