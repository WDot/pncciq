#!/bin/bash

EXPNAME='reproduction'
RESULTSNAME='reproduction_results'
EPOCHS=100

NUM_SAMPLES=16384
LR=1e-4
TIME_SCALE=30
FREQ_SCALE=0.2
SEED=100
LOSS='mse'
NUM_HYPOTHESES=16384
for TASK in "td" "ds" "cf"; do
    BATCH_SIZE=256
    LOSS="mse"
    TRANSFORM="identity"
    python3 main_cc.py --exp_name="${EXPNAME}_${TASK}_${LOSS}_${TRANSFORM}" --batch_size="${BATCH_SIZE}" --lr="${LR}" --num_samples="${NUM_SAMPLES}"  --epochs="${EPOCHS}"  --time_scale="${TIME_SCALE}" --freq_scale="${FREQ_SCALE}" --seed="${SEED}" --task="${TASK}" --loss="${LOSS}" --transform="${TRANSFORM}"
    BATCH_SIZE=1
    python3 cc_sweep.py --exp_name="${RESULTSNAME}_${TASK}_${LOSS}_${TRANSFORM}" --batch_size="${BATCH_SIZE}"  --num_samples="${NUM_SAMPLES}"  --time_scale="${TIME_SCALE}" --seed="${SEED}" --task="${TASK}" --transform="${TRANSFORM}" --denoiser_exp="${EXPNAME}_${TASK}_${LOSS}_${TRANSFORM}" --time_scale="${TIME_SCALE}" --freq_scale="${FREQ_SCALE}" --num_hypotheses="${NUM_HYPOTHESES}"
    BATCH_SIZE=256
    LOSS="mse"
    TRANSFORM="neural"
    python3 main_cc.py --exp_name="${EXPNAME}_${TASK}_${LOSS}_${TRANSFORM}" --batch_size="${BATCH_SIZE}" --lr="${LR}" --num_samples="${NUM_SAMPLES}"  --epochs="${EPOCHS}"  --time_scale="${TIME_SCALE}" --freq_scale="${FREQ_SCALE}" --seed="${SEED}" --task="${TASK}" --loss="${LOSS}" --transform="${TRANSFORM}"
    BATCH_SIZE=1
    python3 cc_sweep.py --exp_name="${RESULTSNAME}_${TASK}_${LOSS}_${TRANSFORM}" --batch_size="${BATCH_SIZE}"  --num_samples="${NUM_SAMPLES}"  --time_scale="${TIME_SCALE}" --seed="${SEED}" --task="${TASK}" --transform="${TRANSFORM}" --denoiser_exp="${EXPNAME}_${TASK}_${LOSS}_${TRANSFORM}" --time_scale="${TIME_SCALE}" --freq_scale="${FREQ_SCALE}" --num_hypotheses="${NUM_HYPOTHESES}"
    BATCH_SIZE=256
    LOSS="cc"
    TRANSFORM="neural"
    python3 main_cc.py --exp_name="${EXPNAME}_${TASK}_${LOSS}_${TRANSFORM}" --batch_size="${BATCH_SIZE}" --lr="${LR}" --num_samples="${NUM_SAMPLES}"  --epochs="${EPOCHS}"  --time_scale="${TIME_SCALE}" --freq_scale="${FREQ_SCALE}" --seed="${SEED}" --task="${TASK}" --loss="${LOSS}" --transform="${TRANSFORM}"
    BATCH_SIZE=1
    python3 cc_sweep.py --exp_name="${RESULTSNAME}_${TASK}_${LOSS}_${TRANSFORM}" --batch_size="${BATCH_SIZE}"  --num_samples="${NUM_SAMPLES}"  --time_scale="${TIME_SCALE}" --seed="${SEED}" --task="${TASK}" --transform="${TRANSFORM}" --denoiser_exp="${EXPNAME}_${TASK}_${LOSS}_${TRANSFORM}" --time_scale="${TIME_SCALE}" --freq_scale="${FREQ_SCALE}" --num_hypotheses="${NUM_HYPOTHESES}"
done