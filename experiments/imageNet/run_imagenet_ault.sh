

ARCH=resnet18
DATASET=imagenet
EPOCHS=90
batch_sizes=(64)
LREPOCHS=30


seeds=(512)


#-------------- Baselines --------------
for SEED in "${seeds[@]}"; do
  for BS in "${batch_sizes[@]}"; do
    sbatch job_ault.sbatch "python3 imagenet_train_eval.py -a ${ARCH} --epochs ${EPOCHS} --batch-size ${BS} \
                --multiprocessing-distributed --workers 8  --world-size 1 --rank 0  \
                /scratch/talbn/imagenet"
done done


#Continue from the checkpoint
#checkpoint_list=(checkpoints/driven-serenity-1810 checkpoints/grateful-terrain-1809 checkpoints/charmed-yogurt-1808)
#for CHECKPINT_DIR in "${checkpoint_list[@]}"; do
#  sbatch job_ault.sbatch "python3 imagenet_train_eval.py -a ${ARCH} --epochs ${EPOCHS} --resume ${CHECKPINT_DIR}  \
#                 --multiprocessing-distributed --world-size 1 --rank 0  \
#                /scratch/talbn/imagenet"
#done

#sbatch job_ault.sbatch "python3 imagenet_train_eval_ault.py -a ${ARCH} --epochs ${EPOCHS} --resume_from_ault  \
#              --multiprocessing-distributed --world-size 1 --rank 0  /scratch/talbn/imagenet"
