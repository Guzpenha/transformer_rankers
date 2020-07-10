cd ../../
mkdir data

mkdir data/qqp
cd data/qqp
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KAFO5l7H89zuNcSQrH08JvcD5bM7S2A_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KAFO5l7H89zuNcSQrH08JvcD5bM7S2A_" -O quora-question-pairs.zip && rm -rf /tmp/cookies.txt
unzip quora-question-pairs.zip
unzip train.csv.zip
rm quora-question-pairs.zip  sample_submission.csv.zip test.csv.zip train.csv.zip test.csv

cd ../../
mkdir data/linkso
cd data/linkso
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X5GoVi_OcRxahXH1pRW7TSesZUeMH3ss' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X5GoVi_OcRxahXH1pRW7TSesZUeMH3ss" -O linkso.tar.gz && rm -rf /tmp/cookies.txt
tar xvf linkso.tar.gz

cd ../../transformer_rankers/scripts
python preprocess_sqr_datasets.py