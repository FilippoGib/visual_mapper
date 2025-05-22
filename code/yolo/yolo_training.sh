#!/bin/bash
#SBATCH --job-name=yolo_training_gibertini
#SBATCH --output=yolo_train.out
#SBATCH --error=yolo_train.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cv

DATA_PATH="/work/cvcs2025/gibertini_gombia_nels/datasets/fsoco/data.yaml"
MODEL="yolov8s.pt"
OUTPUT_DIR=/homes/fgibertini/yolo/yolo_outputs

yolo detect train model=$MODEL data=$DATA_PATH epochs=100 imgsz=800 batch=8 device=0 workers=4 project=$OUTPUT_DIR save_period=10 translate=0.1 scale=0.2 shear=2 hsv_h=0.02 hsv_s=0.02 hsv_v=0.6 mixup=0.02 copy_paste=0.02
