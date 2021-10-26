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

echo "*bart_base_CDS eval"
python3 bart_bias_evaluation.py --model 'debias_methods/saved_models/CDS/CDS_base_bart_1.0_test_8_8_2e-5' 

echo "*bart_large_CDS eval"
python3 bart_bias_evaluation.py --model 'debias_methods/saved_models/CDS/CDS_large_bart_1.0_test_8_2_2e-5' 

echo "*bart_base_ER eval"
python3 bart_bias_evaluation.py --model 'debias_methods/saved_models/ER/ER_base_bart_test_8_8_2e-05_1.25' 

# echo "*bart_large_ER eval"
# python3 bart_bias_evaluation.py --model 'debias_methods/saved_models/ER' 
