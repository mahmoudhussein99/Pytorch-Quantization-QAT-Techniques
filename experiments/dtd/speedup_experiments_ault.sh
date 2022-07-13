ARCH=resnet18

seeds=(512 1024 2048)

#for SEED in "${seeds[@]}"; do
#  sbatch job_ault.sbatch  "python3  tl_train_eval.py  --seed ${SEED} --model ${ARCH}"
#done

act_schemes=(pact)
weight_schemes=(dorefa_weight)
error_scheme=(absmax)
error_roundings=(stochastic)

first_weight_schemes=(lsq_weight)
first_error_scheme=(adaptive)

first_act_scheme=(pact)


for SEED in "${seeds[@]}"; do
  for CONV_AS in "${act_schemes[@]}"; do
    for CONV_WS in "${weight_schemes[@]}"; do
      for CONV_ES in "${error_scheme[@]}"; do
        for CONV_ER in "${error_roundings[@]}"; do
#
          for FIRST_WS in "${first_weight_schemes[@]}"; do
            for FIRST_ES in "${first_error_scheme[@]}"; do

              for FIRST_AS in "${first_act_scheme[@]}"; do

            echo job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --model ${ARCH} \
            --act_qmode ${CONV_AS} --act_bits 4 --weight_qmode ${CONV_WS} --weight_bits 4 \
            --error_qmode ${CONV_ES} --error_rounding ${CONV_ER} --error_man 0 --error_sig 3 --error_rep rdx2 \
            --bn RangeBN --bn_act_bits 16  --bn_error_sig 15 --bn_error_man 0 --bn_weight_bits 16 --bn_act_qmode ${CONV_AS} --bn_weight_qmode ${CONV_WS} --bn_error_qmode ${CONV_ES}\
            --shortcut_quant True \
            --first_weight_qmode ${FIRST_WS} --first_weight_bits 4 --first_error_qmode ${FIRST_ES} --first_error_sig 3 --first_error_man 0 --first_error_rep rdx2 --first_error_rounding stochastic \
            --first_act_qmode ${FIRST_AS} --first_act_bits 8
            "

               done done done
             done done
            done done
          done