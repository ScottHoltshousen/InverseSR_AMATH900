#!/bin/bash
#SBATCH --mail-user=s2holtsh@uwaterloo.ca
#SBATCH --mail-type=begin,end,fail
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100_80g
#SBATCH --job-name="InverseSR_t1"
#SBATCH --time=01:00:00
#SBATCH --output=../bashout/memory_test1/stdout-%j.log
#SBATCH --error=../bashout/memory_test1/stderr-%j.log

SUBJECT_ID=069
cp ~/AMATH900/Research_Project/InverseSR/pytorch_models/IXI_T1_069.pth ../inputs/

cd ../

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

START_STEPS=0
NUM_STEPS=1
LEARNING_RATE=7e-2
LAMBDA_PERC=1e4
CORRUPTION=mask
PRIOR_EVERY=15
PRIOR_AFTER=45
DATA_FORMAT=pth
MASK_ID=9
DOWNSAMPLE_FACTOR=8
N_SAMPLES=3
DDIM_NUM_TIMESTEPS=17
K=1
DDIM_ETA=0.0
EXPERIMENT_NAME=memory_test1
LOG_DIR=~/AMATH900/Research_Project/InverseSR/results/$EXPERIMENT_NAME/tensorboard

python3 ~/AMATH900/Research_Project/InverseSR/project/BRGM_ddim.py \
    --k=$K \
    --ddim_eta=$DDIM_ETA \
    --ddim_num_timesteps=$DDIM_NUM_TIMESTEPS \
    --update_latent_variables \
    --update_conditioning \
    --mean_latent_vector \
    --update_gender \
    --update_age \
    --update_brain \
    --update_ventricular \
    --prior_every=$PRIOR_EVERY \
    --prior_after=$PRIOR_AFTER \
    --num_steps="$NUM_STEPS" \
    --data_format="$DATA_FORMAT" \
    --n_samples="$N_SAMPLES" \
    --subject_id="$SUBJECT_ID" \
    --corruption="$CORRUPTION" \
    --mask_id="$MASK_ID" \
    --lambda_perc="$LAMBDA_PERC" \
    --learning_rate=$LEARNING_RATE \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor="$DOWNSAMPLE_FACTOR" \
    --tensor_board_logger="$LOG_DIR"

zip -r ~/AMATH900/Research_Project/InverseSR/results/$EXPERIMENT_NAME/$EXPERIMENT_NAME.zip results/$EXPERIMENT_NAME
