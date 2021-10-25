#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --qos=gpgpumse
#SBATCH -A punim1431
#SBATCH --time=0-50:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
## Use an account that has GPGPU access

module load fosscuda/2019b
module load pytorch/1.5.1-python-3.7.4
module load tensorflow/2.3.1-python-3.7.4

echo "*bart_base_ER"
python3 bart_ER.py --model 'facebook/bart-base' --lambda_for_ER 1.25 --epochs 8 --lr 2e-5 --batch_size 8 --eval_type test

echo "*bart_large_ER"
python3 bart_ER.py --model 'facebook/bart-large' --lambda_for_ER 1.25 --epochs 8 --lr 2e-5 --batch_size 2 --eval_type test

