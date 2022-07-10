
ARCH=tiny_transformer_v2
DATASET_DIR=fairseq/data-bin/iwslt14.tokenized.de-en

seeds=(512)
fc_act_schemes=(dorefa_act)
fc_weight_schemes=(lsq_weight)
FC_ABIT=4
FC_WBIT=4


ff_error_scheme=(adaptive)
ff_error_roundings=(nearest)
FF_EREP=rdx2
FF_EEXP=3
FF_EMAN=0


mha_linear_act_schemes=(lsq_act)
mha_linear_weight_schemes=(lsq_weight)
MHA_LINEAR_ABIT=8
MHA_LINEAR_WBIT=8

mha_linear_error_schemes=(absmax)
MHA_LIN_EREP=fp
MHA_LIN_ER=nearest
MHA_LIN_EEXP=5
MHA_LIN_EMAN=2


last_layer_weight_schemes=(lsq_weight)
LAST_WBIT=8
last_layer_error_schemes=(adaptive)
LAST_EREP=fp
LAST_ER=nearest
LAST_EEXP=5
LAST_EMAN=2

last_layer_act_schemes=(lsq_act)
LAST_ABIT=8

mha_act_schemes=(pact)
MHA_ABIT=8

mha_error_schemes=(adaptive)
MHA_EREP=fp
MHA_ER=nearest
MHA_EEXP=5
MHA_EMAN=2

 for SEED in "${seeds[@]}"; do
   for FC_AS in "${fc_act_schemes[@]}"; do
     for FC_WS in "${fc_weight_schemes[@]}"; do
       for FF_ES in "${ff_error_scheme[@]}"; do
         for FF_ER in "${ff_error_roundings[@]}"; do

           for MHA_LINEAR_AS in "${mha_linear_act_schemes[@]}"; do
             for MHA_LINEAR_WS in "${mha_linear_weight_schemes[@]}"; do

               for MHA_LIN_ES in  "${mha_linear_error_schemes[@]}"; do

                 for LAST_WS in "${last_layer_weight_schemes[@]}"; do
                   for LAST_ES in "${last_layer_error_schemes[@]}"; do

                   for LAST_AS in "${last_layer_act_schemes[@]}"; do
                   for MHA_AS in "${mha_act_schemes[@]}"; do

                  for MHA_ES in "${mha_error_schemes[@]}"; do

     SAVEDIR="checkpoints/${ARCH}_${FC_AS}_${FC_ABIT}_${FC_WS}_${FC_WBIT}_${FF_ES}_${FF_ER}_${FF_EREP}_${FF_EEXP}_${FF_EMAN}_MHA_LIN_${MHA_LINEAR_AS}_${MHA_LINEAR_ABIT}_${MHA_LINEAR_WS}_${MHA_LINEAR_WBIT}_${MHA_LIN_ES}_${MHA_LIN_ER}_${MHA_LIN_EREP}_${MHA_LIN_EEXP}_${MHA_LIN_EMAN}_LAST_${LAST_WS}_${LAST_WBIT}_${LAST_ES}_${LAST_ER}_${LAST_EREP}_${LAST_EEXP}_${LAST_EMAN}_${LAST_AS}_${LAST_ABIT}_MHA_${MHA_AS}_${MHA_ABIT}_${MHA_ES}_${MHA_ER}_${MHA_EREP}_${MHA_EEXP}_${MHA_EMAN}_seed_${SEED}"
     CMD="${ARCH} ${SEED} ${SAVEDIR} ${DATASET_DIR} ${FC_AS} ${FC_ABIT} ${FC_WS} ${FC_WBIT} ${FF_ES} ${FF_ER} ${FF_EREP} ${FF_EEXP} ${FF_EMAN} ${MHA_LINEAR_AS} ${MHA_LINEAR_ABIT} ${MHA_LINEAR_WS} ${MHA_LINEAR_WBIT} ${MHA_LIN_ES} ${MHA_LIN_ER} ${MHA_LIN_EREP} ${MHA_LIN_EEXP} ${MHA_LIN_EMAN} ${LAST_WS} ${LAST_WBIT} ${LAST_ES} ${LAST_ER} ${LAST_EREP} ${LAST_EEXP} ${LAST_EMAN} ${LAST_AS} ${LAST_ABIT} ${MHA_AS} ${MHA_ABIT} ${MHA_ES} ${MHA_ER} ${MHA_EREP} ${MHA_EEXP} ${MHA_EMAN}"
     echo ${CMD}
     sbatch job_ault.sbatch $CMD
     done done done done done done done done done done done done done