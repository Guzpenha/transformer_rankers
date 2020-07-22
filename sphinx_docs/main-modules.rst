Intro to Components
======================================================

The main components of the library are the following.

datasets
***********

Stores processors for specific datasets as well as code to generate pytorch datasets To download the datasets use *scripts/download_\<task>_data.sh*. Currently implemented processors: 

- **conversation response ranking**: MANtIS, MSDialog, Ubuntu from DSTC8.
- **similar question retrieval**: Quora Question Pair and LinkSO.
- **passage retrieval**: TREC 2020 Passage Ranking.
- **clarifying question retrieval**: ClariQ.

Note that since we choose the negative sampling on the go, we do not read the negative samples from the datasets, only the relevant query-document combinations.


negative_samplers
***********

Currently there is support to query for negative samples using the following approaches:
- **Random**: Selects a random document.
- **BM25**: Uses pyserini to do the retrieval with BM25. Requires anserini installation, follow the *Getting Started* section of their README.
- **sentenceBERT**: Uses sentence embeddings to calculate dense representations of the query and candidates, and faiss is used to do fast retrieval, i.e. dense similarity computation.

See /examples/negative_sampling_example.py for an usage example of the negative samplers.

eval
***********
Uses trec_eval through pytrec_eval library to support most IR evaluation metrics, such as NDCG, MAP, MRR, etc. Additional metrics are implemented here, such as Recall_with_n_candidates@K.

trainers
***********
Transformer trainer supports encoder-only transformers, e.g. BERT, and also encoder-decoder transformers, e.g. T5, from the huggingface transformers library, see their pre-trained models.