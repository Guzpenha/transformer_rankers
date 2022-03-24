Code to reproduce the results of the paper "Sparse and Dense Approaches for the Full-rank Retrieval of Responses for Dialogues".

**Setup**

The following will clone the repo, install a virtual env and install the library with the requirements.

```bash
git clone https://github.com/Guzpenha/transformer_rankers.git
cd transformer_rankers
python3 -m venv env
source env/bin/activate
pip install -e .
pip install -r requirements.txt
```

Two files of the sentence-transformers (https://github.com/UKPLab/sentence-transformers/) were changed:  RerankingEvaluator.py and SentenceTransformer.py. The changes are for getting the best model overall using the MAP value and logging with wandb. Update both files on the enviroment folder of the sentence-transformer library, using the files from ./scripts/sentence_transformers_mod/.

Installing anserini is also required:

```bash
apt-get install maven -qq
git clone --recurse-submodules https://github.com/castorini/anserini.git
cd anserini; mvn clean package appassembler:assemble -DskipTests -Dmaven.javadoc.skip=true
ls anserini/target/appassembler/bin/
```

**Scripts to reproduce results**

In order to reproduce the results of the paper the following scripts should be used:

- Sparse response expansion: train_and_apply_resp2context.sh
- Sparse dialogue context expansion: fine_tune_rm3.sh, eval_random_rm3.sh
- Zero-shot dense models: eval_zeroshot_dense.sh
- Fine-tuned dense models: train_and_apply_dense_models_for_crr.sh
