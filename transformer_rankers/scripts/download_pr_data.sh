cd ../../
mkdir data

mkdir data/trec2020pr
cd data/trec2020pr
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv
tar xvf collection.tar.gz
tar xvf queries.tar.gz