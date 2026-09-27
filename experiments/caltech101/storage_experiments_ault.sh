ARCH=resnet18

seeds=(512 1024 2048)

#for SEED in "${seeds[@]}"; do
#  sbatch job_ault.sbatch  "python3  cifar_train_eval.py  --seed ${SEED} --model ${ARCH}"
#done

bn_act_schemes=(pact)
conv_act_schemes=(pact)


for SEED in "${seeds[@]}"; do
  for BN_AS in "${bn_act_schemes[@]}"; do
    for CONV_AS in "${conv_act_schemes[@]}"; do

            sbatch job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --model ${ARCH} \
            --bn_act_qmode ${BN_AS} --bn_act_bits 4 \
            --act_qmode ${CONV_AS} --act_bits 4 \
            --shortcut_quant True
            "


            sbatch job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --model ${ARCH} \
            --bn_act_qmode ${BN_AS} --bn_act_bits 4 \
            --act_qmode ${CONV_AS} --act_bits 4 \
            --shortcut_quant True \
            --optim_qmode bnb
            "

done
  done
    done