#!/bin/bash
#SBATCH --job-name=yolo_training_gibertini
#SBATCH --output=yolo_train.out
#SBATCH --error=yolo_train.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load cuda/11.8
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cv

DATA_PATH="/work/cvcs2025/gibertini_gombia_nels/datasets/fsoco/data.yaml"
MODEL="yolov8n.pt"
OUTPUT_DIR=/homes/fgibertini/yolo/yolo_outputs

yolo detect train model=$MODEL data=$DATA_PATH epochs=10 imgsz=640 batch=4 device=0 workers=2 project=$OUTPUT_DIR
