from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_metric, load_dataset

from transformer_rankers.datasets import downloader
from IPython import embed
from tqdm import tqdm


import pandas as pd
import argparse
import logging
import os
import torch


def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder to save the processed data")
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="dataset")
    parser.add_argument("--t5_model", default=None, type=str, required=True,
                        help="the t5 model for response2context")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    def preprocess_response(r):
        # some tokens that only appear in MSDialogue are removed here
        r = r.replace("<<<AGENT>>>:", "")
        r = r.replace("PERSON_PLACEHOLDER", "")
        r = r.replace("AGENT", "")
        print(r)
        return r
    
    tasks = [args.task] 

    for task in tasks:
        for split in ['train', 'test']: #['train', 'valid', 'test']:
            print("Applying on {}/{}".format(task, split))
            data = pd.read_csv(args.data_folder+task+"/{}.tsv".format(split), sep="\t")

            data["response"] = data.apply(lambda r,f=preprocess_response: f(r['response']), axis=1)

            model_checkpoint = "t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            inputs = data['response'].tolist()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_inputs = tokenizer(inputs,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            model = T5ForConditionalGeneration.from_pretrained(args.t5_model).to(device)

            batch_size = 16
            predictions = []
            for start in tqdm(range(0, model_inputs['input_ids'].shape[0], batch_size)):
                batch_inputs = {'input_ids': model_inputs['input_ids'][start:start+batch_size]}
                batch_inputs['input_ids'] = batch_inputs['input_ids'].to(device)
                preds_batch = model.generate(
                    **batch_inputs,
                    max_length=400,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=3)                
                preds_batch_decoded = tokenizer.batch_decode(
                    preds_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for s in range(0, len(preds_batch_decoded), 3):
                    predictions.append(preds_batch_decoded[s]+ " [AUG] " + preds_batch_decoded[s+1] + " [AUG] " + preds_batch_decoded[s+2])
            os.makedirs(args.data_folder + task + "_resp2context_last_utt_{}/".format("last_utt" in args.t5_model), exist_ok=True)
            predictions = pd.DataFrame([pred.strip() for pred in predictions],columns=["generated_context"])
            original_data = pd.read_csv(args.data_folder+task+"/{}.tsv".format(split), sep="\t")
            final_df = original_data.join(predictions)
            final_df["response"] = final_df.apply(lambda r: r["response"] + " [AUG] " + r["generated_context"],axis=1)
            final_df[['context', 'response']].to_csv(args.data_folder + task + "_resp2context_last_utt_{}/".
                format("last_utt" in args.t5_model)+split+".tsv", index=False, sep='\t')

if __name__ == "__main__":
    main()