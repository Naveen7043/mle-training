
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.logger.logging import logging

try:

    def load_data(data_path):
        logging.info("Loading data in dataframe in train module")
        return pd.read_csv(data_path)

    def preprocess_features(data):
        logging.info("Preprocessing initizated")
        housing = data.drop("median_house_value", axis=1)
        housing_labels = data["median_house_value"]
        num_attribs = list(housing.select_dtypes(include=[np.number]))
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

        housing_prepared = full_pipeline.fit_transform(housing)
        logging.info(housing_prepared)
        logging.info(housing_labels)
        return housing_prepared, housing_labels

    def train_linear_regression(housing_prepared, housing_labels):
        logging.info("Creating Linear Regression model")
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        return lin_reg

    def train_decision_tree(housing_prepared, housing_labels):
        logging.info("Creating Decision Tree model")
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)
        return tree_reg

    def train_random_forest(housing_prepared, housing_labels):
        logging.info("Creating Random forest model")
        forest_reg = RandomForestRegressor(random_state=42)
        forest_reg.fit(housing_prepared, housing_labels)
        return forest_reg

    def randomized_search_cv(housing_prepared, housing_labels):
        logging.info("Initizating Random Search cv")
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        logging.info(rnd_search.best_estimator_)
        return rnd_search.best_estimator_

    def train_and_save_models(train_data_path, model_output_path):
        data = load_data(train_data_path)
        housing_prepared, housing_labels = preprocess_features(data)

        # Train multiple models
        logging.info("Training models with the given data")
        models = {
            "linear_regression": train_linear_regression(housing_prepared,
                                                         housing_labels),
            "decision_tree": train_decision_tree(housing_prepared,
                                                 housing_labels),
            "random_forest": train_random_forest(housing_prepared,
                                                 housing_labels),
            "random_forest_tuned": randomized_search_cv(housing_prepared,
                                                        housing_labels),
        }

        # Save models
        for model_name, model in models.items():
            logging.info("Model trained and saved at f{}".
                        # format(model_output_path))
            joblib.dump(model, f"{model_output_path}/{model_name}.pkl")

except Exception as e:
    raise CustomException(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Train multiple models and save them."))
    parser.add_argument(
        "train_data_path", type=str, help="Path to the training data CSV file."
    )
    parser.add_argument(
        "model_output_path", type=str,
        help="Path to save the trained model files."
    )
    args = parser.parse_args()
    train_and_save_models(args.train_data_path, args.model_output_path)
