# DA-Gender
This is the Github repo for DA-Gender, a challenge dataset for measuring the extent of over-debiasing.
# The Dataset
introduce 
and domains
# Set up
Use Python 3 (we use Python 3.7) and install the required packages.
```
pip install -r requirements.txt
```
# bias mitigation
We took BERT as an examples:
-Counterfactual Data Substitution (CDS)
```
python3 bert_CDS.py --model 'bert-base-uncased' --CDS_ratio 1.0 --epochs 8 --lr 2e-5 --batch_size 8 --eval_type test
```
-Embedding Regularization (ER)
```
python3 bert_ER.py --model 'bert-base-uncased' --lambda_for_ER 0.5 --epochs 8 --lr 2e-5 --batch_size 8 --eval_type test 
```
-Sentence Debias (SD)
```
python3 bert_sent_debias.py --model 'bert-base-uncased' --target_pair_ratio 0.05 --eval_type test
```

# bias and over-debiasing evaluation
For each model, there is a notebook that shows the bias and over-debiasing evaluation results before and after debiased by SD.
Or you can run evaluation by the .py file.
Again, we took BERT as an example:

-Evaluate orginal models
```
python3 bert_bias_evaluation.py --model "bert-base-uncased"
```
-Evaluate models that debiased by ER/CDS
```
python3 bert_bias_evaluation.py --debiased_model_path "bert-base-uncased" 
```
-Evaluate models that debiased by SD
```
python3 bert_bias_evaluation.py --model "bert-base-uncased" --subspace_path
```
