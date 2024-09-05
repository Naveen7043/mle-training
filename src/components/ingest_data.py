import argparse
import os
import tarfile

import pandas as pd
from six.moves import urllib  # type: ignore
from sklearn.model_selection import train_test_split

# from src.exception.exception import CustomException
# from src.logger.logging import logging

try:

    '''def fetch_housing_data(housing_url, housing_path):
#       logging.info("Fetching data from give URL")
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()'''
    def fetch_housing_data(housing_url, housing_path):
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        with tarfile.open(tgz_path) as housing_tgz:
            housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    def load_housing_data(housing_path):
#        logging.info("Loading data into Dataframe")
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def split_data(data, test_size, seed):
#        logging.info("Spiltting data into train and test")
        return train_test_split(data, test_size=test_size, random_state=seed)

except Exception as e:
    raise Exception(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Fetch and split housing data."))
    parser.add_argument(
        "output_folder", type=str,
        help="Output folder for the raw and split data."
    )
    args = parser.parse_args()

    DOWNLOAD_ROOT = """
    https://raw.githubusercontent.com/ageron/handson-ml/master/"""
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    HOUSING_PATH = os.path.join(args.output_folder, "housing")

    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)
    train_set, test_set = split_data(housing, test_size=0.2, seed=42)
    train_set.to_csv(os.path.join(args.output_folder, "train.csv"),
                     index=False)
    test_set.to_csv(os.path.join(args.output_folder, "test.csv"), index=False)
