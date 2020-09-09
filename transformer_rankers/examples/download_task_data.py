from transformer_rankers.datasets import downloader
from IPython import embed

import pandas as pd
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to download [{}]".format(",".join(downloader.TASK_TO_URLS.keys())))
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder to save the processed data")
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.info("Starting downloader for task {}".format(args.task))
    dataDownloader = downloader.DataDownloader(args.task, args.data_folder)
    dataDownloader.download_and_preprocess()

if __name__ == "__main__":
    main()