
ARCH=resnet18
DATASET=cifar10
LREPOCH=50
EPOCHS=150



seeds=(512 1025 2048)
# 4096 8192)

#Baselines
#for SEED in "${seeds[@]}"; do
#  sbatch job_ault.sbatch  "python3  cifar_train_eval.py  --seed ${SEED} --model resnet50"
#done



act_rep=(int)
act_bits=(8)
act_schemes=(dorefa_act pact lsq_act)

weight_rep=(int)
weight_bits=(8)
weight_schemes=(lsq_weight sawb dorefa_weight)


##8-bit Forward Quantization
for SEED in "${seeds[@]}"; do
  for AS in "${act_schemes[@]}"; do
    for WS in "${weight_schemes[@]}"; do
          sbatch job_ault.sbatch "python3 cifar_train_eval.py --lr 0.01 --seed ${SEED} --act_qmode ${AS} --act_bits 8  --weight_qmode ${WS} --weight_bits 8  --shortcut_quant False --model ${ARCH}"
            done done done



error_scheme=(fixed absmax)
error_roundings=(nearest stochastic)


##With 4-bit Error quantization
#for SEED in "${seeds[@]}"; do
#  for ES in "${error_scheme[@]}"; do
#    for ER in "${error_roundings[@]}"; do
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 --error_qmode ${ES} --error_rounding ${ER} --error_man 0 --error_sig 3 --error_rep rdx2 --shortcut_quant False --error_scale 1000000.0 --model ${ARCH}"
#            done done done

#With 8-bit Error quantization
#for SEED in "${seeds[@]}"; do
#    for ES in "${error_scheme[@]}"; do
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --act_qmode lsq_act --act_bits 8  --weight_qmode dorefa_weight --weight_bits 8 --error_qmode ${ES} --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --shortcut_quant False --error_scale 1000000.0 --model ${ARCH}"
#          done done



ES=absmax
ER=stochastic
bn_weight_schemes=(lsq_weight)
bn_act_schemes=(pact)
bn_error_scheme=(absmax)







#Original BN (4, 8, 16):
#for SEED in "${seeds[@]}"; do
#  for BNWS in "${bn_weight_schemes[@]}"; do
#    for BNAS in "${bn_act_schemes[@]}"; do
#      for BNES in "${bn_error_scheme[@]}"; do
##          sbatch job_ault.sbatch "python3 cifar_train_eval.py   --bn_weight_bits 4 --bn_act_bits 4 --bn_error_man 0 --bn_error_sig 3 --bn_error_rep fp --bn_error_rounding stochastic  --bn_weight_qmode dorefa_weight --bn_act_qmode lsq_act --bn_error_qmode fixed  --seed ${SEED} --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 --error_qmode fixed  --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --shortcut_quant False --bn OriginalRangeBN  --model ${ARCH} --error_scale 1000000.0"
#
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py   --bn_weight_bits 8 --bn_act_bits 8 --bn_error_man 2 --bn_error_sig 5 --bn_error_rep fp --bn_error_rounding stochastic  --bn_weight_qmode dorefa_weight --bn_act_qmode lsq_act --bn_error_qmode fixed  --seed ${SEED} --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 --error_qmode fixed  --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --shortcut_quant False --bn OriginalRangeBN  --model ${ARCH} --error_scale 1000000.0"
#
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py   --bn_weight_bits 16 --bn_act_bits 16 --bn_error_man 10 --bn_error_sig 5 --bn_error_rep fp --bn_error_rounding stochastic  --bn_weight_qmode dorefa_weight --bn_act_qmode lsq_act --bn_error_qmode fixed  --seed ${SEED} --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 --error_qmode fixed  --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --shortcut_quant False --bn OriginalRangeBN  --model ${ARCH} --error_scale 1000000.0 "
#done done done done




#Original BN (16) with Shortcut Quant:
#for SEED in "${seeds[@]}"; do
#  for BNWS in "${bn_weight_schemes[@]}"; do
#    for BNAS in "${bn_act_schemes[@]}"; do
#      for BNES in "${bn_error_scheme[@]}"; do
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 16 --bn_act_bits 16 --bn_error_man 10 --bn_error_sig 5 --bn_error_rep fp --bn_error_rounding nearest  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} --seed ${SEED} --act_qmode pact --act_bits 4  --weight_qmode lsq_weight --weight_bits 4 --error_qmode ${ES} --error_rounding ${ER} --error_man 0 --error_sig 3 --error_rep rdx2 --shortcut_quant True --bn OriginalRangeBN  --model ${ARCH}"
#done done done done

# BN (FP32) with Shortcut Quant (focused on first layer):
#for SEED in "${seeds[@]}"; do
#
#          # 4-bits first layer:
##          sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} \
##          --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 \
##          --error_qmode fixed --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --error_scale 1000000.0 \
##          --first_act_qmode lsq_act --first_act_bits 4  --first_weight_qmode dorefa_weight --first_weight_bits 4 --first_error_qmode fixed --first_error_rounding stochastic --first_error_man 0 --first_error_sig 3 --first_error_rep rdx2 \
##           --shortcut_quant True  --model ${ARCH}  "
##
##          # 8-bits first layer:
##          sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} \
##          --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 \
##          --error_qmode fixed --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --error_scale 1000000.0 \
##          --first_act_qmode lsq_act --first_act_bits 8  --first_weight_qmode dorefa_weight --first_weight_bits 8 --first_error_qmode fixed --first_error_rounding nearest --first_error_man 5 --first_error_sig 2 --first_error_rep fp \
##           --shortcut_quant True  --model ${ARCH}  "
#
#          # 16-bits first layer:
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} \
#          --act_qmode lsq_act --act_bits 4  --weight_qmode dorefa_weight --weight_bits 4 \
#          --error_qmode fixed --error_rounding nearest --error_man 2 --error_sig 5 --error_rep fp --error_scale 1000000.0 \
#          --first_act_qmode lsq_act --first_act_bits 16  --first_weight_qmode dorefa_weight --first_weight_bits 16 --first_error_qmode fixed --first_error_rounding nearest --first_error_man 5 --first_error_sig 10 --first_error_rep fp \
#           --shortcut_quant True  --model ${ARCH}  "
#
#
#
#done









#-----------------------------------------------------------------------------------------------------Storage-----------------------------------------------------------------

act_schemes=(dorefa_act pact lsq_act)
#4-bit Activation Quantization
#for SEED in "${seeds[@]}"; do
#  for AS in "${act_schemes[@]}"; do
#
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --model mobilenet \
#          --act_qmode ${AS} --act_bits 4  --shortcut_quant False "
#
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py  --seed ${SEED} --model resnet18 \
#          --act_qmode ${AS} --act_bits 4  --shortcut_quant False "
#
#done done




#First step: BN quant forward:
#Original BN with Shortcut Quant:
# for SEED in "${seeds[@]}"; do
#   for BNWS in "${bn_weight_schemes[@]}"; do
#     for BNAS in "${bn_act_schemes[@]}"; do
#       for BNES in "${bn_error_scheme[@]}"; do
#           sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant False --model ${ARCH}\
#           --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 16 --bn_act_bits 16 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding nearest  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#           --act_qmode lsq_act --act_bits 4  --bn OriginalRangeBN"
#
#           sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant False --model ${ARCH}\
#           --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 8 --bn_act_bits 8 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding nearest  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#           --act_qmode lsq_act --act_bits 4  --bn OriginalRangeBN"
#
#           sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant False --model ${ARCH}\
#           --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 4 --bn_act_bits 4 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding nearest  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#           --act_qmode lsq_act --act_bits 4  --bn OriginalRangeBN"
# done done done done



#Last two lines of the table!
# for SEED in "${seeds[@]}"; do
#   for BNWS in "${bn_weight_schemes[@]}"; do
#     for BNAS in "${bn_act_schemes[@]}"; do
#       for BNES in "${bn_error_scheme[@]}"; do
#
#           sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant True --model ${ARCH}\
#           --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 8 --bn_act_bits 8 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding nearest  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#           --act_qmode lsq_act --act_bits 4  --bn OriginalRangeBN"
#
#           sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant True --model ${ARCH} --optim_qmode bnb\
#           --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 8 --bn_act_bits 8 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding nearest  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#           --act_qmode lsq_act --act_bits 4  --bn OriginalRangeBN"
#
#
# done done done done


#for SEED in "${seeds[@]}"; do
#  for BNWS in "${bn_weight_schemes[@]}"; do
#    for BNAS in "${bn_act_schemes[@]}"; do
#      for BNES in "${bn_error_scheme[@]}"; do
##          With bnb optimizer without shortcut
#
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant True --model ${ARCH} --optim_qmode bnb\
#          --bn OriginalRangeBN --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 8 --bn_act_bits 8 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding stochastic  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#          --act_qmode pact --act_bits 4  --weight_qmode lsq_weight --weight_bits 4 \
#          --first_act_qmode pact --first_act_bits 4  --first_weight_qmode lsq_weight --first_weight_bits 4
#          "
#
#          sbatch job_ault.sbatch "python3 cifar_train_eval.py --seed ${SEED} --shortcut_quant True --model ${ARCH} --optim_qmode bnb\
#          --bn OriginalRangeBN --bn_act_rounding nearest --bn_weight_rounding nearest  --bn_weight_bits 8 --bn_act_bits 8 --bn_error_man 8 --bn_error_sig 23 --bn_error_rep fp --bn_error_rounding stochastic  --bn_weight_qmode ${BNWS} --bn_act_qmode ${BNAS} --bn_error_qmode ${BNES} \
#          --act_qmode pact --act_bits 4  --weight_qmode lsq_weight --weight_bits 4 \
#          --first_act_qmode pact --first_act_bits 8  --first_weight_qmode lsq_weight --first_weight_bits 8
#          "
#
#done done done done
