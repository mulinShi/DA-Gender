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

echo "*bert_base_sents"
python3 bert_sent_debias.py --model 'bert-base-uncased' --target_pair_ratio 0.05 --eval_type test

echo "*bert_large_sents"
python3 bert_sent_debias.py --model 'bert-large-uncased' --target_pair_ratio 0.05 --eval_type test