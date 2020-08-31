cd transformer_rankers/scripts/

#Download data and train sentenceBERT
./download_crr_data.sh
nohup ./run_sentence_BERT_crr.sh > train_sentence_BERT_1_epoch.log &

#Run models for CRR on cross_task and cross_ns setups.
nohup ./run_crr_ensemble.sh > ensemble.log &
nohup ./run_crr_MC_dropout.sh > mc_dropout.log & 

#Aggregate ensemble runs and analyze the results using 
#aggregate_ensemble_files.py and analyze_uncertainty.py