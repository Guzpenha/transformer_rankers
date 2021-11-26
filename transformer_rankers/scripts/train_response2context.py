from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from datasets import load_metric, load_dataset, Dataset

from transformer_rankers.datasets import downloader
from IPython import embed
from tqdm import tqdm

import pandas as pd
import argparse
import logging
import os

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder to save the processed data")
    parser.add_argument("--last_utterance_only", default=False, required=False, action="store_true",
                        help="Train with the whole context or the last utterance only")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    tasks = ['mantis', 'msdialog', 'ubuntu_dstc8']
    #Downloading Conversation Response Ranking
    for task in tasks:
        if not os.path.isdir(args.data_folder+task):
            logging.info("Starting downloader for task {}".format(task))
            dataDownloader = downloader.DataDownloader(task, args.data_folder)
            dataDownloader.download_and_preprocess()
    
    all_df = []
    for task in tasks:
        train = pd.read_csv(args.data_folder+task+"/train.tsv", sep="\t")
        train['task'] = task
        replace = train.shape[0]<80000
        train = train.sample(80000, replace=replace)
        all_df.append(train)
    all_df = pd.concat(all_df)

    def preprocess_response(r):
        # some tokens that only appear in MSDialogue are removed here
        r = r.replace("<<<AGENT>>>:", "")
        r = r.replace("PERSON_PLACEHOLDER", "")
        r = r.replace("AGENT", "")
        print(r)
        return r

    def preprocess_context(r):
        #removes beginning of context and keeps only last utterance.
        return r.split("[TURN_SEP]")[-1].split("[UTTERANCE_SEP]")[0].strip()

    all_df["response"] = all_df.apply(lambda r,f=preprocess_response: f(r['response']), axis=1)
    if args.last_utterance_only:
        all_df["context"] = all_df.apply(lambda r,f=preprocess_context: f(r['context']), axis=1)

    dataset = Dataset.from_pandas(all_df)

    # all_df["len_context"] = all_df.apply(lambda r: len(r['context'].split(" ")), axis=1)
    # all_df["len_response"] = all_df.apply(lambda r: len(r['response'].split(" ")), axis=1)

    model_checkpoint = "t5-base" #["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    max_input_length = 100
    if args.last_utterance_only:
        max_target_length = 100    
    else:
        max_target_length = 400

    col_from = "response"
    col_to = "context"

    def preprocess_function(examples):
        inputs = [preprocess_response(doc) for doc in examples[col_from]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples[col_to]], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 5
    train_args = Seq2SeqTrainingArguments(
        "response2context_lu_{}".format(args.last_utterance_only),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,        
        predict_with_generate=True,
        seed=42
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    print("Fine-tuning T5.")
    trainer.train()
    if args.last_utterance_only:
        model.save_pretrained("{}/{}_response2context_last_utt_only".format(args.data_folder, model_checkpoint))
    else:
        model.save_pretrained("{}/{}_response2context".format(args.data_folder, model_checkpoint))

if __name__ == "__main__":
    main()