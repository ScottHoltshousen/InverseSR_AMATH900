#!/bin/bash
#SBATCH --mail-user=s2holtsh@uwaterloo.ca
#SBATCH --mail-type=begin,end,fail
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100_80g
#SBATCH --job-name="InverseSR"
#SBATCH --time=02:00:00
#SBATCH --output=../bashout/IXI_T1_069/downsample/cond/update_false/stdout-%j.log
#SBATCH --error=../bashout/IXI_T1_069/downsample/cond/update_false/stderr-%j.log

DATASPEC=IXI_T1_069
SUBJECT_ID=069
TESTTYPE=cond
#cp ~/AMATH900/Research_Project/InverseSR/data/IXI022-Guys-0701-T1.nii.gz ../inputs/

cd ../

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"

START_STEPS=0
NUM_STEPS=500
LEARNING_RATE=7e-2
LAMBDA_PERC=5e4
CORRUPTION=downsample
PRIOR_EVERY=15
PRIOR_AFTER=45
DATA_FORMAT=pth
MASK_ID=5
DOWNSAMPLE_FACTOR=4
N_SAMPLES=3
DDIM_NUM_TIMESTEPS=15
K=1
DDIM_ETA=0.0
EXPERIMENT_DIR=$DATASPEC/$CORRUPTION/$TESTTYPE
EXPERIMENT_NAME=update_false
LOG_DIR=~/AMATH900/Research_Project/InverseSR/results/$EXPERIMENT_DIR/$EXPERIMENT_NAME/tensorboard

python3 ~/AMATH900/Research_Project/InverseSR/project/BRGM_ddim.py \
    --k=$K \
    --ddim_eta=$DDIM_ETA \
    --ddim_num_timesteps=$DDIM_NUM_TIMESTEPS \
    --update_latent_variables="True" \
    --update_conditioning="False" \
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
    --experiment_dir=$EXPERIMENT_DIR \
    --experiment_name=$EXPERIMENT_NAME \
    --downsample_factor="$DOWNSAMPLE_FACTOR" \
    --tensor_board_logger="$LOG_DIR"

zip -r ~/AMATH900/Research_Project/InverseSR/results/$EXPERIMENT_DIR/$EXPERIMENT_NAME/$EXPERIMENT_NAME.zip results/$EXPERIMENT_DIR/$EXPERIMENT_NAME
