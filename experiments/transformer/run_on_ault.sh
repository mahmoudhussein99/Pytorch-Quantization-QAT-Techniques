
seeds=(512 1025 2048 4096 8192)
ARCH=tiny_transformer_v2
DATASET_DIR=fairseq/data-bin/iwslt14.tokenized.de-en

#Baselines
#for SEED in "${seeds[@]}"; do
#    BASELINEDIR="checkpoints/${ARCH}_Baseline_seed_${SEED}"
#    BASELINECMD="${ARCH} ${SEED} ${BASELINEDIR} ${DATASET_DIR}"
#    echo $BASELINECMD
#    sbatch job_ault.sbatch $BASELINECMD
#done


