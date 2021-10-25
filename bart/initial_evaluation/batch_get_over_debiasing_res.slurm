#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --qos=gpgpumse
#SBATCH -A punim1431
#SBATCH --time=0-30:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
## Use an account that has GPGPU access

module load fosscuda/2019b
module load pytorch/1.5.1-python-3.7.4
module load tensorflow/2.3.1-python-3.7.4

echo "Bias evaluation"
python3 batch_get_over_debiasing_res.py --models_dir '../get_debiased_models/base_models/' --debiasing_type "CDS" --output_file 'bart_base_over_debiasing_res/'
python3 batch_get_over_debiasing_res.py --models_dir '../get_debiased_models/base_models/' --debiasing_type "ER"  --output_file 'bart_base_over_debiasing_res/'

python3 batch_get_over_debiasing_res.py --models_dir '../get_debiased_models/large_models/' --debiasing_type "CDS" --output_file 'bart_large_over_debiasing_res/'
python3 batch_get_over_debiasing_res.py --models_dir '../get_debiased_models/large_models/' --debiasing_type "ER" --output_file 'bart_large_over_debiasing_res/'



