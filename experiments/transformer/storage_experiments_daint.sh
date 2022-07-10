
ARCH=tiny_transformer_v2
DATASET_DIR=fairseq/data-bin/iwslt14.tokenized.de-en

seeds=(512)

OPTIM=bnb
mha_act_schemes=(pact)
MHA_AB=4

ff_act_schemes=(pact)
FF_AB=4

mha_linear_act_schemes=(pact)
MHA_LIN_ABIT=4

normalization_act_schemes=(pact dorefa_act lsq_act)
NORM_ACTB=8

for SEED in "${seeds[@]}"; do
  for MHA_AS in "${mha_act_schemes[@]}"; do
    for FF_AS in "${ff_act_schemes[@]}"; do
      for MHA_LIN_AS in  "${mha_linear_act_schemes[@]}"; do
        for NORM_AS in  "${normalization_act_schemes[@]}"; do
    SAVEDIR="checkpoints/${ARCH}_${OPTIM}_MHA_${MHA_AS}_${MHA_AB}_FF_${FF_AS}_${FF_AB}_MHA_LINEAR_${MHA_LIN_AS}_${MHA_LIN_ABIT}_NORM_${NORM_AS}_${NORM_ACTB}_seed_${SEED}"

    CMD="${ARCH} ${SEED} ${SAVEDIR} ${DATASET_DIR} ${OPTIM} ${MHA_AS} ${MHA_AB} ${FF_AS} ${FF_AB} ${MHA_LIN_AS} ${MHA_LIN_ABIT} ${NORM_AS} ${NORM_ACTB}"
    echo ${CMD}
    sbatch job_daint.sbatch $CMD
    done
    done
    done
    done
    done


