cd ../../
mkdir data

mkdir data/mantis
cd data/mantis
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nf_JRR7zIcCLrzvL_vRsuzBxDcD_3g6N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nf_JRR7zIcCLrzvL_vRsuzBxDcD_3g6N" -O response_ranking_dataset_easy.7z && rm -rf /tmp/cookies.txt
7za e response_ranking_dataset_easy.7z
mv data_dev_easy.tsv valid.tsv
mv data_test_easy.tsv test.tsv
mv data_train_easy.tsv train.tsv
rm response_ranking_dataset_easy.7z
rm data_dev_easy_lookup.txt
rm data_test_easy_lookup.txt
rm data_train_easy_lookup.txt

cd ../../
mkdir data/msdialog
cd data/msdialog
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R_c8b7Yi0wChA_du3eKDtnOGuYTqVhnY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R_c8b7Yi0wChA_du3eKDtnOGuYTqVhnY" -O MSDialog.tar.gz && rm -rf /tmp/cookies.txt
tar -xzf MSDialog.tar.gz
mv MSDialog/train.tsv .
mv MSDialog/test.tsv .
mv MSDialog/valid.tsv .
rm MSDialog.tar.gz
rm -rf MSDialog

cd ../../
mkdir data/ubuntu_dstc8
cd data/ubuntu_dstc8