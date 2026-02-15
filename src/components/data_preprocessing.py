import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.outliers import OutlierCapper

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataPreprocessingConfig:
    train_processed_path: str = os.path.join('artifacts', 'processed', 'train.csv')
    test_processed_path: str = os.path.join('artifacts', 'processed', 'test.csv')
    preprocessor_path: str = os.path.join('artifacts', 'preprocesser.pkl')

class DataPreprocessor:
    def __init__(self):
        self.preprocessor_config = DataPreprocessingConfig()

    def initiate_preprocessing(self, df_train, df_test):
        try:
            logging.info('Data Preprocessing started.')
            # drop the id column from training and testing dataset
            df_train = df_train.drop('id', axis=1)
            df_test = df_test.drop('id', axis=1)

            # extracting the column names
            features = df_train.columns.to_list()
            features.remove('MedHouseVal') 

            not_normal_feats = ~df_train.skew().between(-0.5, 0.5)
            iqr_feats = not_normal_feats[not_normal_feats].index.tolist()
            # removing the MedHouseVal feature from iqr_feats
            iqr_feats.remove('MedHouseVal')

            # preprocessing pipeline
            pipeline = Pipeline([
                ('capper', OutlierCapper(iqr_feats)),
                ('scaler', StandardScaler())
            ])

            # separating independent and dependent features from training dataset
            X_train = df_train.drop('MedHouseVal', axis=1)
            y_train = df_train['MedHouseVal']

            X_test = df_test.drop('MedHouseVal', axis=1)
            y_test = df_test['MedHouseVal']

            # transforming the datasets
            X_train_processed = pipeline.fit_transform(X_train)
            X_test_processed = pipeline.transform(X_test)

            # log transform MedHouseVal feature
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)

            # converting the processed datasets into dataframes
            df_train_processed = pd.concat([
                pd.DataFrame(X_train_processed, columns=X_train.columns),
                pd.DataFrame(y_train, columns=['MedHouseVal'])
            ], axis=1)

            df_test_processed = pd.concat([
                pd.DataFrame(X_test_processed, columns=X_test.columns),
                pd.DataFrame(y_test, columns=['MedHouseVal'])
            ], axis=1)

            logging.info('Datasets preprocessed successfully.')

            # saving the processed datasets
            os.makedirs(os.path.dirname(self.preprocessor_config.train_processed_path), exist_ok=True)

            df_train_processed.to_csv(self.preprocessor_config.train_processed_path, index=False, header=True)
            df_test_processed.to_csv(self.preprocessor_config.test_processed_path, index=False, header=True)

            # saving the preprocessing pipeline 
            with open(self.preprocessor_config.preprocessor_path, 'wb') as f:
                pickle.dump(pipeline, f)

            logging.info('Preprocessing completed.')

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':

    # path for the training and testing datasets
    train_data_path = 'artifacts/raw/train.csv'
    test_data_path = 'artifacts/raw/test.csv'

    # loading the datasets
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    # initiating data preprocessing
    data_preprocessor = DataPreprocessor()
    data_preprocessor.initiate_preprocessing(df_train, df_test)