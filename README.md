# DA-Gender
This is the Github repo for DA-Gender, a challenge dataset for measuring the extent of over-debiasing.
# The Dataset
DA-Gender measures the degree of genuine (or: desirable) associations that are present in PLMs. It consists of 2610 pairs of sentences, one encoding a factual/genuine association (She is pregnant), and the other a violation of the fact (He is pregnant). Each example has the following information:
- Sent_A: The sentence that contains desirable associations
- Sent_B: The sentence that voilates desirable associations in Sent_A.
- TM_sent: The sentence with the target word masked.
- TAM_sent: The sentence with the attribute word masked.
- Used_tmp: The template for constructing the sentence pair.
- Attr: The attribute word in the sentence pair.
- Male_target: The male target word.
- Female_target: The female target word.
- ideal_pref: The label which specifies the gender of the attribute word.

# Set up
Use Python 3 (we use Python 3.7) and install the required packages.
```
pip install -r requirements.txt
```
# bias mitigation
We took BERT-base as an examples:
- Counterfactual Data Substitution (CDS)
```
python3 bert_CDS.py 
          --model 'bert-base-uncased' 
          --CDS_ratio [The hyperparameter for controlling the debiasing effects of CDS] 
          --epochs [The number of fine-tuning epoch ]
          --lr [The learning rate] 
          --batch_size [The batch size]
          --eval_type [Which groups of data for bias evaluation (val/test/whole set)]
```
- Embedding Regularization (ER)
```
python3 bert_ER.py 
          --model 'bert-base-uncased' 
          --lambda_for_ER [The hyperparameter for controlling the debiasing effects of ER] 
          --epochs [The number of fine-tuning epoch ]
          --lr [The learning rate] 
          --batch_size [The batch size]
          --eval_type [Which groups of data for bias evaluation (val/test/whole set)] 
```
- Sentence Debias (SD)
```
python3 bert_sent_debias.py 
          --model 'bert-base-uncased' 
          --target_pair_ratio [The hyperparameter for controlling the debiasing effects of SD] 
          --eval_type [Which groups of data for bias evaluation (val/test/whole set)] 
```

# bias and over-debiasing evaluation
For each model, there is a notebook that shows the bias and over-debiasing evaluation results before and after debiased by SD.
Or you can run evaluation by the .py file.
Again, we took BERT as an example:

- Evaluate orginal models
```
python3 bert_bias_evaluation.py --model "bert-base-uncased"
```
- Evaluate models that debiased by ER/CDS
```
python3 bert_bias_evaluation.py --debiased_model_path "bert-base-uncased" 
```
- Evaluate models that debiased by SD
```
python3 bert_bias_evaluation.py --model "bert-base-uncased" --subspace_path
```
