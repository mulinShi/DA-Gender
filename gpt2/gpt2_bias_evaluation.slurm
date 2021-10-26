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

echo "*GPT2_base_CDS eval"
python3 gpt2_bias_evaluation.py --model 'debias_methods/saved_models/CDS/CDS__gpt2_0.9_test_8_8_5e-05' 

echo "*GPT2_medium_CDS eval"
python3 gpt2_bias_evaluation.py --model 'debias_methods/saved_models/CDS/CDS_medium_gpt2_0.9_test_8_2_5e-05' 

echo "*GPT2_base_ER eval"
python3 gpt2_bias_evaluation.py --model 'debias_methods/saved_models/ER/ER_gpt2__test_8_8_5e-05_0.5' 

echo "*GPT2_medium_ER eval"
python3 gpt2_bias_evaluation.py --model 'debias_methods/saved_models/ER/ER_gpt2_medium_test_8_2_5e-05_1.0' 
