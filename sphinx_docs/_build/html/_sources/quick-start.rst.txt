Quick-start
======================================================

Setup
***********
1. Clone the repository:
::
    git clone git@github.com:Guzpenha/transformer_rankers.git    
    cd transformer_rankers

2. [Optional] Create a virtual env (python >= 3.6) and activate it:
::
   python3 -m venv env
   source env/bin/activate

3. Install the library:
::
    pip install -e .
   
4. Install the requirements:
::
   pip install -r requirements.txt

Example
***********

1. Download and preprocess Similar Question Retrieval data:
::
   cd transformer_rankers/scripts
   ./download_sqr_data.sh

2. Train BERT-ranker for Quora Question Pairs (with only 1000 samples to be fast):
::
   python ../examples/crr_bert_ranker_example.py \
         --task qqp \
         --data_folder ../data/ \
         --output_dir ../data/output_data \
         --sample_data 1000

The output will be something like this:
:: 
   [...]
   2020-06-23 11:19:44,522 [INFO] Epoch 1 val nDCG@10: 0.245
   2020-06-23 11:19:44,522 [INFO] Predicting
   2020-06-23 11:19:44,523 [INFO] Starting evaluation on test.
   2020-06-23 11:20:03,678 [INFO] Test ndcg_cut_10: 0.3236

3. The experiment info will be saved at *../data/output_data*, where you can find the following files:
::

   /data/output_data/1/config.json
   /data/output_data/1/cout.txt
   /data/output_data/1/labels.csv
   /data/output_data/1/predictions.csv
   /data/output_data/1/run.json

4. You can easily aggregate the results of different experiment runs using */examples/crr_results_analyses_example.py*:
