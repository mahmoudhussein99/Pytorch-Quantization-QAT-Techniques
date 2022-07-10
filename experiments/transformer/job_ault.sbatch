#!/bin/bash -l
#
#SBATCH --job-name=Tformer_Training
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --partition=amdv100,intelv100,amda100
#SBATCH --exclude=ault10
#SBATCH --constraint=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --account=sashkboo

export PYTHONPATH="${PYTHONPATH}:/users/sashkboo/State-of-Quantization-in-DL"
source ../../env/bin/activate

ARCH=$1
SEED=$2
SAVEDIR=$3
DATASET_DIR=$4


FC_AS=$5
FC_ABIT=$6
FC_WS=$7
FC_WBIT=$8

FF_ES=$9
FF_ER=${10}
FF_EREP=${11}
FF_EEXP=${12}
FF_EMAN=${13}

MHA_LINEAR_AS=${14}
MHA_LINEAR_ABIT=${15}
MHA_LINEAR_WS=${16}
MHA_LINEAR_WBIT=${17}


MHA_LINEAR_ES=${18}
MHA_LINEAR_ER=${19}
MHA_LINEAR_EREP=${20}
MHA_LINEAR_EEXP=${21}
MHA_LINEAR_EMAN=${22}


LAST_WS=${23}
LAST_WBIT=${24}
LAST_ES=${25}
LAST_ER=${26}
LAST_EREP=${27}
LAST_EEXP=${28}
LAST_EMAN=${29}

LAST_AS=${30}
LAST_AB=${31}



MHA_AS=${32}
MHA_ABIT=${33}


MHA_ES=${34}
MHA_ER=${35}
MHA_EREP=${36}
MHA_EEXP=${37}
MHA_EMAN=${38}


srun python3 transformer_train_eval.py ${DATASET_DIR} --arch ${ARCH}  --seed ${SEED} \
      --distributed-world-size 1 \
      --nprocs-per-node 1 \
       --device-id $(( $RANDOM % 4)) \
      --save-dir ${SAVEDIR} --tensorboard-logdir ${SAVEDIR} \
      --share-decoder-input-output-embed \
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
      --fc_act_qmode ${FC_AS} --fc_act_bits ${FC_ABIT} --fc_weight_qmode ${FC_WS} --fc_weight_bits ${FC_WBIT} \
      --fc_error_qmode ${FF_ES} --fc_error_rep ${FF_EREP} --fc_error_rounding ${FF_ER} --fc_error_sig ${FF_EEXP} --fc_error_man ${FF_EMAN} \
      --mha_linear_act_qmode ${MHA_LINEAR_AS} --mha_linear_act_bits ${MHA_LINEAR_ABIT}  --mha_linear_weight_qmode ${MHA_LINEAR_WS} --mha_linear_weight_bits ${MHA_LINEAR_WBIT} \
      --mha_linear_error_qmode ${MHA_LINEAR_ES} --mha_linear_error_rep ${MHA_LINEAR_EREP} --mha_linear_error_rounding ${MHA_LINEAR_ER} --mha_linear_error_sig ${MHA_LINEAR_EEXP} --mha_linear_error_man ${MHA_LINEAR_EMAN} \
      --last_weight_qmode ${LAST_WS} --last_weight_bits ${LAST_WBIT} --last_error_qmode ${LAST_ES} --last_error_rep ${LAST_EREP}  --last_error_rounding ${LAST_ER}  --last_error_sig ${LAST_EEXP}  --last_error_man ${LAST_EMAN} \
      --last_act_qmode ${LAST_AS} --last_act_bits ${LAST_AB} \
      --mha_act_qmode ${MHA_AS} --mha_act_bits ${MHA_ABIT} \
      --mha_error_qmode ${MHA_ES} --mha_error_rep ${MHA_EREP}  --mha_error_rounding ${MHA_ER}  --mha_error_sig ${MHA_EEXP}  --mha_error_man ${MHA_EMAN}
