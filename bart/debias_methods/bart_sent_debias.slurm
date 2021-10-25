#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --qos=gpgpumse
#SBATCH -A punim1431
#SBATCH --time=0-99:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
## Use an account that has GPGPU access

module load fosscuda/2019b
module load pytorch/1.5.1-python-3.7.4
module load tensorflow/2.3.1-python-3.7.4

echo "*sent_debias bart_base"
python3 bart_sent_debias.py --model 'facebook/bart-base' --target_pair_ratio 0.07 --eval_type test

echo "*sent_debias bart_large"
python3 bart_sent_debias.py --model 'facebook/bart-large' --target_pair_ratio 0.07 --eval_type test