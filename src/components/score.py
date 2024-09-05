import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.logger.logging import logging
from src.exception.exception import CustomException


def load_model(model_path):
    logging.info("Loading Model")
    return joblib.load(model_path)


def preprocess_features(data):
    logging.info("Preprocessing file in test")
    num_attribs = list(data.select_dtypes(include=[np.number]))
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )
    housing_prepared = full_pipeline.fit_transform(data)
    return housing_prepared


def score_models(model_path, data_path, output_path):
    logging.info("Model Scoring Began")
    data = pd.read_csv(data_path)
    features = preprocess_features(data.drop("median_house_value", axis=1))
    Check if model_path is a directory or a single file
    if os.path.isdir(model_path):
        models = {
            name: load_model(os.path.join(model_path, name))
            for name in os.listdir(model_path)
            if name.endswith(".pkl")
        }
    else:
        model_name = os.path.basename(model_path)
        models = {model_name: load_model(model_path)}
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory {output_path}")
    # Predict and save results
    for name, model in models.items():
        predictions = model.predict(features)
        output_file_path = os.path.join(output_path, f"{name}_predictions.csv")
        pd.DataFrame(predictions,
                     columns=["Predictions"]).to_csv(output_file_path)
        logging.info(f"Predictions saved to {output_file_path}")
        print(f"Predictions saved to {output_file_path}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Score a model or models and save predictions."
        )
        parser.add_argument(
            "model_path", type=str,
            help="Path to the trained model file or folder."
        )
        parser.add_argument(
            "data_path", type=str,
            help="Path to the dataset CSV file for scoring."
        )
        parser.add_argument(
            "output_path", type=str,
            help="Path to save the prediction output files."
        )
        args = parser.parse_args()
        score_models(args.model_path, args.data_path, args.output_path)
    except Exception as e:
        raise CustomException(e, sys)
