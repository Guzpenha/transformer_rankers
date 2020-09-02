from transformer_rankers.datasets import preprocess_crr

def main():
    data_path = "../../data/ubuntu_dstc8"
    train = preprocess_crr.read_crr_tsv_as_df(data_path+"/train.tsv", -1, add_turn_separator=False)
    valid = preprocess_crr.read_crr_tsv_as_df(data_path+"/valid.tsv", -1, add_turn_separator=False)
    
    train.to_csv(data_path+"/train.tsv", index=False, sep="\t")
    valid[0:int(valid.shape[0]/2)].to_csv(data_path+"/valid.tsv", index=False, sep="\t")
    valid[int(valid.shape[0]/2):].to_csv(data_path+"/test.tsv", index=False, sep="\t")

    data_path = "../../data/mantis"
    train = preprocess_crr.read_crr_tsv_as_df(data_path+"/train.tsv")
    valid = preprocess_crr.read_crr_tsv_as_df(data_path+"/valid.tsv")
    test = preprocess_crr.read_crr_tsv_as_df(data_path+"/test.tsv")
    train.to_csv(data_path+"/train.tsv", index=False, sep="\t")
    valid.to_csv(data_path+"/valid.tsv", index=False, sep="\t")
    test.to_csv(data_path+"/test.tsv", index=False, sep="\t")

    data_path = "../../data/msdialog"
    train = preprocess_crr.read_crr_tsv_as_df(data_path+"/train.tsv")
    valid = preprocess_crr.read_crr_tsv_as_df(data_path+"/valid.tsv")
    test = preprocess_crr.read_crr_tsv_as_df(data_path+"/test.tsv")
    train.to_csv(data_path+"/train.tsv", index=False, sep="\t")
    valid.to_csv(data_path+"/valid.tsv", index=False, sep="\t")
    test.to_csv(data_path+"/test.tsv", index=False, sep="\t")

if __name__ == "__main__":
    main()