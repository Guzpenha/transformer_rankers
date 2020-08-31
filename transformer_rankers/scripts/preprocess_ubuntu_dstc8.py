from transformer_rankers.datasets.preprocess_crr import transform_dstc8_to_tsv

def main():
    data_path = "../../data/ubuntu_dstc8"
    for f_path, f_o_path in [("{}/task-1.ubuntu.dev.json".format(data_path), "{}/valid.tsv".format(data_path)),
                            ("{}/task-1.ubuntu.train.json".format(data_path), "{}/train.tsv".format(data_path))]:
        print("transforming {}".format(f_path))
        data = transform_dstc8_to_tsv(f_path)
        if 'dev' in f_path:
            data_dev, data_test = data[0:len(data)//2], data[len(data)//2:]
            with open(f_o_path, 'w') as f_write:
                for l in data_dev:
                    f_write.write(l)
            with open("{}/test.tsv".format(data_path), 'w') as f_write:
                for l in data_test:
                    f_write.write(l)
        else:
            with open(f_o_path, 'w') as f_write:
                for l in data:
                    f_write.write(l)

if __name__ == "__main__":
    main()