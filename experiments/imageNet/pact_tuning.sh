
pact_reg=(0.00002)


#Our Storage Setup:
for PACT_REG in "${pact_reg[@]}"; do

sbatch job_ault.sbatch "python3 imagenet_slurm.py /users/sashkboo/ImageNet-100-Pytorch/imageNet100 --batch-size 256 -a resnet50  --epochs 40  --pact_reg ${PACT_REG} --find_unused_param True \
               --dist-backend nccl --multiprocessing-distributed --workers 8 \
              --act_qmode pact --act_bits 4 --weight_qmode sawb --weight_bits 4   \
            --error_qmode absmax --error_rounding stochastic --error_man 0 --error_sig 3 --error_rep rdx2 \
            --first_weight_qmode lsq_weight --first_weight_bits 4 --first_error_qmode adaptive --first_error_sig 3 --first_error_man 0 --first_error_rep rdx2 --first_error_rounding stochastic      \
            --first_act_qmode pact --first_act_bits 8
             "
done



#sbatch job_ault.sbatch "python3 imagenet_slurm.py /users/sashkboo/ImageNet-100-Pytorch/imageNet100 --batch-size 201 -a resnet50  --epochs 40   --find_unused_param True \
#               --dist-backend nccl --multiprocessing-distributed --workers 8 \
#            --act_qmode lsq_act --act_bits 4 --weight_qmode dorefa_weight --weight_bits 4   \
#            --error_qmode absmax --error_rounding stochastic --error_man 0 --error_sig 3 --error_rep rdx2      \
#            --bn RangeBN --bn_act_bits 16  --bn_error_sig 15 --bn_error_man 0 --bn_weight_bits 16 --bn_act_qmode lsq_act --bn_weight_qmode dorefa_weight --bn_error_qmode absmax    \
#            --first_weight_qmode lsq_weight --first_weight_bits 4 --first_error_qmode adaptive --first_error_sig 3 --first_error_man 0 --first_error_rep rdx2 --first_error_rounding stochastic      \
#            --first_act_qmode lsq_act --first_act_bits 8"


#sbatch job_ault.sbatch "python3 imagenet_slurm.py /users/sashkboo/ImageNet-100-Pytorch/imageNet100 --batch-size 256 -a resnet50  --epochs 40  --find_unused_param True \
#               --dist-backend nccl --multiprocessing-distributed --workers 8 \
#            --act_qmode pact --act_bits 4 --weight_qmode dorefa_weight --weight_bits 4   \
#            --error_qmode absmax --error_rounding stochastic --error_man 0 --error_sig 3 --error_rep rdx2      \
#            --first_weight_qmode lsq_weight --first_weight_bits 4 --first_error_qmode adaptive --first_error_sig 3 --first_error_man 0 --first_error_rep rdx2 --first_error_rounding stochastic      \
#            --first_act_qmode pact --first_act_bits 8"