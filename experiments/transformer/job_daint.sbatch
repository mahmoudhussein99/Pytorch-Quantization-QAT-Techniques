#!/bin/bash -l
#
#SBATCH --job-name=tformer_storage
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=tformer-%j.out
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=g34

module load daint-gpu
source /scratch/snx3000/sashkboo/State-of-Quantization-in-DL/env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/scratch/snx3000/sashkboo/State-of-Quantization-in-DL"


ARCH=$1
SEED=$2
SAVEDIR=$3
DATASET_DIR=$4

OPTIM=$5

MHA_AS=$6
MHA_AB=$7

FF_AS=$8
FF_AB=$9
MHA_LIN_AS=${10}
MHA_LIN_ABIT=${11}

NORM_AS=${12}
NORM_ACTB=${13}

srun python3 transformer_train_eval.py ${DATASET_DIR} --arch ${ARCH}  --seed ${SEED} \
      --distributed-world-size 1 \
      --nprocs-per-node 1 \
      --save-dir ${SAVEDIR} --tensorboard-logdir ${SAVEDIR} \
      --share-decoder-input-output-embed \
      --optimizer ${OPTIM} \
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
      --update-freq 1 \
      --no-progress-bar \
      --log-format simple \
      --log-interval 100 \
      --save-interval-updates 1000 \
      --keep-interval-updates 20  \
      --eval-bleu \
      --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
      --eval-bleu-detok moses \
      --eval-bleu-remove-bpe \
      --eval-bleu-print-samples \
      --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
      --wandb-project state_quantization_neurips22 \
      --device-id $(( $RANDOM % 4)) \
      --task qtranslation \
      --max-epoch 50 \
      --mha_act_qmode ${MHA_AS} --mha_act_bits ${MHA_AB} \
      --fc_act_qmode ${FF_AS} --fc_act_bits ${FF_AB} \
      --mha_linear_act_qmode ${MHA_LIN_AS} --mha_linear_act_bits ${MHA_LIN_ABIT} \
      --ln_act_qmode ${NORM_AS} --ln_act_bits ${NORM_ACTB}