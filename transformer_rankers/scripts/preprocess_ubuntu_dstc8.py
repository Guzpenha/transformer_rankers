from transformer_rankers.datasets.preprocess_crr import transform_dstc8_to_tsv

data_path = "../../data/ubuntu_dstc8"
for f_path, f_o_path in [("{}/task-1.ubuntu.dev.json".format(data_path), "{}/valid.tsv".format(data_path)),
                         ("{}/task-1.ubuntu.train.json".format(data_path), "{}/train.tsv".format(data_path))]:
    print("transforming {}".format(f_path))
    data = transform_dstc8_to_tsv(f_path)
    with open(f_o_path, 'w') as f_write:
        for l in data:
            f_write.write(l)