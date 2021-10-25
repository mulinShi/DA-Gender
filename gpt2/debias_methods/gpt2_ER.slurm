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

echo "*gpt2_ER"
python3 gpt2_ER.py --lambda_for_ER 0.5 --model 'gpt2' --epochs 8 --lr 5e-5 --batch_size 8 --eval_type test

echo "*gpt2-medium_ER"
python3 gpt2_ER.py --lambda_for_ER 1.0 --model 'gpt2-medium' --epochs 8 --lr 5e-5 --batch_size 2 --eval_type test

