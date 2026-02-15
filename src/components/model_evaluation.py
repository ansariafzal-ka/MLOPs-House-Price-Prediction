import pandas as pd
import sys
import pickle
import mlflow

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

class ModelEvaluation:
    def initiate_model_evaluation(self, model, X_test, y_test):
        try:
            logging.info('Model evaluation started.')
            logging.info('Making predictions on test data...')
            y_pred = model.predict(X_test)

            logging.info('Calculating metrics...')
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.set_experiment('House Price Prediction')
            with mlflow.start_run(run_name='model_evaluation'):
                mlflow.log_metric('mse', mse)
                mlflow.log_metric('mae', mae)
                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('R2 Score', r2)

            logging.info(f'MSE: {mse:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2 Score: {r2:.3f}')
            logging.info('Model evaluation completed.')

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    # path for processed testing dataset
    df_test_processed_path = 'artifacts/processed/test.csv'

    #path for trained model
    model_path = 'artifacts/models/model.pkl'

    # loading the processed testing dataset
    df_test = pd.read_csv(df_test_processed_path)

    # loading the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # separating the independent and dependent variables
    X_test = df_test.drop('MedHouseVal', axis=1)
    y_test = df_test['MedHouseVal']

    # initiating model evaluation
    model_evaluator = ModelEvaluation()
    model_evaluator.initiate_model_evaluation(model, X_test, y_test)