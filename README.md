## Median housing value prediction
The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modeled the median house value on given housing data.
The following techniques have been used:
- Linear regression
- Decision Tree
- Random Forest
## Steps performed
- We prepare and clean the data. We check and impute for missing values.
- Features are generated and the variables are checked for correlation.
- Multiple sampling techinuqies are evaluated. The data set is split into train and test.
- All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.
## To excute the script
- Run these before running the Python script:
- conda env create -f env.yml
- conda activate mle-dev
- pip install -e .
- python src/components/ingest_data.py /mnt/c/Users/sai.mallampally/mle-training/artifacts
- python src/components/train.py artifacts/train.csv artifacts/
- python src/components/score.py artifacts/random_forest_tuned.pkl artifacts/valid.csv artifacts/prediction_validate.csv
- Please make sure the order of exection remains the same to ensure sucessfull execution
